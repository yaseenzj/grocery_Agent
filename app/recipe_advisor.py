from __future__ import annotations
import asyncio
import json
import logging
from pathlib import Path
from typing import Any
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from app.inventory_manager import get_current_inventory
from app.database import get_db_pool
from app.models import InventoryItem, InventoryPayload, Recipe
logger = logging.getLogger(__name__)

_MODEL_NAME = 'gemini-3.5-flash'
_FALLBACK_1 = 'gemini-2.5-flash'
_FALLBACK_2 = 'gemini-1.5-flash'
_TEMPERATURE = 0.0
_MAX_RETRIES = 5
_RECIPE_COUNT = 3

class _MealRecommendationPayload(BaseModel):
    recipes: list[Recipe] = Field(..., description=f'Exactly {_RECIPE_COUNT} complete Recipe objects generated from the provided ingredient list. Every recipe must use at least one ingredient from the inventory list and must include full steps, ingredients, and a source URL.', min_length=1)
_SYSTEM_PROMPT = "You are an expert culinary AI backend assistant for a smart grocery inventory system. Your job is to analyze a list of available ingredients and return a structured list of realistic recipes. Your goal is to generate healthy, realistic recipes using available ingredients.\n\nStrict Composition Constraints:\n\n1. Ingredient Prioritization: Prioritize using the ingredients at the TOP of the provided list, as they expire first and must be consumed before going to waste. Recipes that use more top-ranked ingredients are preferred over those that skip them.\n\n2. Stock Constraints (CRITICAL): NEVER include core ingredients, meats, vegetables, or fruits that are NOT in the provided inventory list. You are strictly limited to the ingredients provided. If you cannot make a full recipe without adding a major missing ingredient, make a simpler recipe instead.\n\n3. Complementary Ingredients: If a recipe needs basic complementary components not present in the inventory list (e.g., olive oil, salt, pepper, spices, flour), you MAY include them — but you MUST list each one in the recipe's `restock_recommendations` array with a precise quantity and unit. Do not silently assume pantry staples are available.\n\n4. Recipe Quality: Each recipe must be complete and realistic:\n   - `recipe_name`: descriptive and human-readable.\n   - `difficulty_level`: string indicating level of cooking (e.g., 'Easy', 'Medium', 'Hard').\n   - `ingredients`: full list with quantities and units.\n   - `steps`: ordered, atomic cooking instructions (minimum 4 steps).\n   - `source`: a valid HTTP/HTTPS URL to a reputable cooking reference.\n   - `restock_recommendations`: list of items needed but not in inventory      (empty list [] if nothing extra is required).\n\n5. Output Format: Return exactly the structured JSON matching the provided schema. Do not emit explanations, markdown fences, or prose outside the JSON object."

def _format_ingredient_block(items: list[InventoryItem]) -> str:
    lines: list[str] = []
    for rank, item in enumerate(items, start=1):
        lines.append(f'{rank:>2}. {item.item_name.title():<30} {item.count} {item.unit:<6}  (expires: {item.expiration_date})')
    return '\n'.join(lines)

def _build_human_message(inventory_block: str) -> str:
    return f'Here are my currently available ingredients, sorted by expiration date (most urgent first):\n\n{inventory_block}\n\nPlease generate {_RECIPE_COUNT} healthy, realistic recipes that make the best possible use of these ingredients, prioritizing those at the top of the list. Follow all composition constraints from the system prompt exactly.'

def _compute_inventory_match_score(recipe: Recipe, available_names: frozenset[str]) -> float:
    if not recipe.ingredients:
        logger.debug("Recipe '%s' has no ingredients — score=0.0", recipe.recipe_name)
        return 0.0
    matched = sum((1 for ing in recipe.ingredients if ing.item_name.strip().lower() in available_names))
    score = round(matched / len(recipe.ingredients), 4)
    logger.debug("Match score | recipe='%s' | matched=%d/%d | score=%.4f", recipe.recipe_name, matched, len(recipe.ingredients), score)
    return score

def _enrich_and_rank_recipes(recipes: list[Recipe], available_names: frozenset[str]) -> list[Recipe]:
    enriched: list[Recipe] = []
    for recipe in recipes:
        score = _compute_inventory_match_score(recipe, available_names)
        enriched.append(recipe.model_copy(update={'inventory_match_score': score}))
    enriched.sort(key=lambda r: r.inventory_match_score or 0.0, reverse=True)
    return enriched

async def recommend_optimized_meals() -> list[Recipe]:
    logger.info('Recipe advisor: loading inventory snapshot.')
    inventory: InventoryPayload = await get_current_inventory()
    active_items: list[InventoryItem] = [it for it in inventory.items if it.count > 0]
    if not active_items:
        logger.warning('Recipe advisor: inventory is empty or fully consumed. Returning empty recipe list — user should be prompted to restock.')
        return []
    logger.info('Recipe advisor: %d active ingredient(s) after filtering (of %d total).', len(active_items), len(inventory.items))
    active_items.sort(key=lambda it: it.expiration_date)
    available_names: frozenset[str] = frozenset((it.item_name.strip().lower() for it in active_items))
    ingredient_block = _format_ingredient_block(active_items)
    human_message_text = _build_human_message(ingredient_block)
    logger.debug('Recipe advisor: sending %d ingredients to LLM.\n%s', len(active_items), ingredient_block)
    try:
        primary = ChatGoogleGenerativeAI(model=_MODEL_NAME, temperature=_TEMPERATURE, max_retries=_MAX_RETRIES)
        fb1 = ChatGoogleGenerativeAI(model=_FALLBACK_1, temperature=_TEMPERATURE, max_retries=_MAX_RETRIES)
        fb2 = ChatGoogleGenerativeAI(model=_FALLBACK_2, temperature=_TEMPERATURE, max_retries=_MAX_RETRIES)
        primary_chain = primary.with_structured_output(_MealRecommendationPayload)
        fb1_chain = fb1.with_structured_output(_MealRecommendationPayload)
        fb2_chain = fb2.with_structured_output(_MealRecommendationPayload)
        structured_llm = primary_chain.with_fallbacks([fb1_chain, fb2_chain])
    except Exception as exc:
        logger.error('Recipe advisor: failed to build LangChain chain | error=%s', exc, exc_info=True)
        raise RuntimeError(f'Could not initialise the LLM chain for recipe generation. Verify that GOOGLE_API_KEY is set and langchain_google_genai is installed. Underlying error: {exc}') from exc
    try:
        messages = [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=human_message_text)]
        result: _MealRecommendationPayload = await structured_llm.ainvoke(messages)
    except ValueError as exc:
        logger.error('Recipe advisor: schema validation failed | error=%s', exc, exc_info=True)
        raise RuntimeError(f'The LLM returned a recipe payload that failed Pydantic validation. The inventory context may have been too complex or the model quota was hit. Validation error: {exc}') from exc
    except Exception as exc:
        logger.error('Recipe advisor: unexpected LLM error | error=%s', exc, exc_info=True)
        raise RuntimeError(f'Recipe generation failed due to an unexpected error: {exc}') from exc
    if not result.recipes:
        logger.warning('Recipe advisor: LLM returned zero recipes despite non-empty inventory. Inventory context preview: %s', ingredient_block[:300])
        return []
    logger.info('Recipe advisor: LLM returned %d recipe(s). Computing match scores.', len(result.recipes))
    ranked_recipes = _enrich_and_rank_recipes(result.recipes, available_names)
    logger.info("Recipe advisor: complete | recipes=%d | top_recipe='%s' | top_score=%.2f", len(ranked_recipes), ranked_recipes[0].recipe_name if ranked_recipes else 'N/A', ranked_recipes[0].inventory_match_score if ranked_recipes else 0.0)
    return ranked_recipes

async def save_cached_recipes(recipes: list[Recipe]) -> None:
    try:
        pool = await get_db_pool()
        payload = json.dumps([r.model_dump() for r in recipes])
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute('DELETE FROM recipes_cache')
                await conn.execute('INSERT INTO recipes_cache (recipes_json) VALUES ($1)', payload)
        logger.info('Saved %d recipes to database cache.', len(recipes))
    except Exception as exc:
        logger.error('Failed to save recipes to cache | error=%s', exc, exc_info=True)

async def precompute_and_cache_recipes() -> None:
    logger.info('Background recipe precomputation started.')
    try:
        recipes = await recommend_optimized_meals()
        await save_cached_recipes(recipes)
        logger.info('Background recipe precomputation complete. Cached %d recipes.', len(recipes))
    except (asyncio.CancelledError, KeyboardInterrupt):
        logger.info('Background recipe precomputation cancelled during shutdown.')
    except Exception as exc:
        logger.error('Background recipe precomputation failed | error=%s', exc, exc_info=True)

async def read_cached_recipes() -> list[Recipe]:
    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow('SELECT recipes_json FROM recipes_cache ORDER BY cached_at DESC LIMIT 1')
            if not row:
                return []
            data = json.loads(row['recipes_json'])
            recipes = []
            for r in data:
                score = r.get('inventory_match_score', 0.0)
                r['inventory_match_score'] = max(0.0, min(1.0, float(score)))
                diff = r.get('difficulty_level', 'Medium')
                if diff not in ['Easy', 'Medium', 'Hard']:
                    r['difficulty_level'] = 'Medium'
                recipes.append(Recipe.model_validate(r))
            return recipes
    except Exception as exc:
        logger.error('Failed to read cached recipes from DB | error=%s', exc)
        return []