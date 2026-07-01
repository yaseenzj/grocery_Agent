"""
app/recipe_advisor.py
─────────────────────────────────────────────────────────────────────────────
Step 4 of the Grocery Lifecycle Pipeline: Prioritized Recipe Advisory Loop.

Reads the live inventory snapshot, filters and sorts ingredients by urgency,
then calls Gemini Pro via LangChain's structured-output binding to receive
fully validated Recipe objects.

Key design decisions:
  - Uses an internal _MealRecommendationPayload wrapper around list[Recipe]
    because .with_structured_output() is more reliably bound to a root
    Pydantic model than to a raw Python generic (list[Recipe]).
  - inventory_match_score is computed locally after the LLM responds —
    this avoids burdening the model with a scoring task and ensures the
    score is always grounded in the actual filtered inventory state.
  - Recipes are returned sorted by inventory_match_score descending so the
    caller always sees the most achievable dish first.

Pipeline position:
  data/inventory.json → recipe_advisor.py → list[Recipe]
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from app.inventory_manager import get_current_inventory
from app.models import InventoryItem, InventoryPayload, Recipe

# ─────────────────────────────────────────────────────────────────────────────
# Logger & Constants
# ─────────────────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)

_RECIPE_PATH = Path(__file__).parent.parent / "data" / "recipes.json"

_MODEL_NAME  = "gemini-3.5-flash"  
_FALLBACK_1  = "gemini-2.5-flash"
_FALLBACK_2  = "gemini-1.5-flash"
_TEMPERATURE = 0.0   
_MAX_RETRIES = 5
_RECIPE_COUNT = 3    


# ─────────────────────────────────────────────────────────────────────────────
# Internal Pydantic Wrapper
# ─────────────────────────────────────────────────────────────────────────────

class _MealRecommendationPayload(BaseModel):
    """
    Internal container used exclusively as the structured-output target.

    Wraps list[Recipe] inside a named field so that Gemini's function-calling
    mechanism has a concrete root object to bind to. Using a raw Python generic
    (list[Recipe]) as the schema target is unreliable across LangChain versions;
    this wrapper guarantees a stable JSON Schema anchor.

    The outer list is unwrapped before returning to the caller of
    recommend_optimized_meals(), so consumers never see this type.
    """

    recipes: list[Recipe] = Field(
        ...,
        description=(
            f"Exactly {_RECIPE_COUNT} complete Recipe objects generated from the "
            "provided ingredient list. Every recipe must use at least one ingredient "
            "from the inventory list and must include full steps, ingredients, and a source URL."
        ),
        min_length=1,
    )


# ─────────────────────────────────────────────────────────────────────────────
# System Prompt
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an advanced culinary optimization engine. Your goal is to generate \
healthy, realistic recipes using available ingredients.

Strict Composition Constraints:

1. Ingredient Prioritization: Prioritize using the ingredients at the TOP of \
the provided list, as they expire first and must be consumed before going to waste. \
Recipes that use more top-ranked ingredients are preferred over those that skip them.

2. Stock Constraints (CRITICAL): NEVER include core ingredients, meats, vegetables, or fruits that are NOT in the provided inventory list. You are strictly limited to the ingredients provided. If you cannot make a full recipe without adding a major missing ingredient, make a simpler recipe instead.

3. Complementary Ingredients: If a recipe needs basic complementary components \
not present in the inventory list (e.g., olive oil, salt, pepper, spices, flour), \
you MAY include them — but you MUST list each one in the recipe's \
`restock_recommendations` array with a precise quantity and unit. \
Do not silently assume pantry staples are available.

4. Recipe Quality: Each recipe must be complete and realistic:
   - `recipe_name`: descriptive and human-readable.
   - `difficulty_level`: string indicating level of cooking (e.g., 'Easy', 'Medium', 'Hard').
   - `ingredients`: full list with quantities and units.
   - `steps`: ordered, atomic cooking instructions (minimum 4 steps).
   - `source`: a valid HTTP/HTTPS URL to a reputable cooking reference.
   - `restock_recommendations`: list of items needed but not in inventory \
     (empty list [] if nothing extra is required).

5. Output Format: Return exactly the structured JSON matching the provided schema. \
Do not emit explanations, markdown fences, or prose outside the JSON object.\
"""


# ─────────────────────────────────────────────────────────────────────────────
# Prompt Formatting Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _format_ingredient_block(items: list[InventoryItem]) -> str:
    """
    Render the filtered, sorted inventory into a numbered prompt block.

    The positional numbering matters: the system prompt instructs the LLM to
    prioritize top-ranked (lower-index) items, which correspond to the
    soonest-expiring ingredients after the sort applied by the caller.

    Example output:
        1. Broccoli — 1.0 pcs  (expires: 2025-07-05)
        2. Chicken Breast — 2.0 lbs  (expires: 2025-07-07)
        3. Butter — 2.0 pcs  (expires: 2025-07-30)

    Args:
        items: Active inventory items sorted by expiration_date ascending.

    Returns:
        A multi-line string ready for injection into the human prompt.
    """
    lines: list[str] = []
    for rank, item in enumerate(items, start=1):
        lines.append(
            f"{rank:>2}. {item.item_name.title():<30} "
            f"{item.count} {item.unit:<6}  "
            f"(expires: {item.expiration_date})"
        )
    return "\n".join(lines)


def _build_human_message(inventory_block: str) -> str:
    """
    Compose the full human turn injected into the LLM conversation.

    Args:
        inventory_block: Pre-formatted ingredient table from
            ``_format_ingredient_block``.

    Returns:
        A fully composed human-turn string.
    """
    return (
        f"Here are my currently available ingredients, sorted by expiration date "
        f"(most urgent first):\n\n"
        f"{inventory_block}\n\n"
        f"Please generate {_RECIPE_COUNT} healthy, realistic recipes that make the "
        f"best possible use of these ingredients, prioritizing those at the top of "
        f"the list. Follow all composition constraints from the system prompt exactly."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Post-Processing Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_inventory_match_score(
    recipe: Recipe,
    available_names: frozenset[str],
) -> float:
    """
    Compute the fraction of a recipe's ingredients already in the inventory.

    The score is a simple coverage ratio:
        matched_ingredients / total_ingredients

    Matching is performed case-insensitively on the normalized item_name.
    A score of 1.0 means all ingredients are on hand; 0.0 means none are.

    This is computed locally (not by the LLM) to guarantee it is always
    grounded in the actual, filtered inventory state at call time.

    Args:
        recipe:          A Recipe returned by the LLM.
        available_names: Frozenset of lowercase item_names from the active
                         inventory (count > 0 entries only).

    Returns:
        Float in [0.0, 1.0], rounded to 4 decimal places.
    """
    if not recipe.ingredients:
        logger.debug("Recipe '%s' has no ingredients — score=0.0", recipe.recipe_name)
        return 0.0

    matched = sum(
        1
        for ing in recipe.ingredients
        if ing.item_name.strip().lower() in available_names
    )
    score = round(matched / len(recipe.ingredients), 4)
    logger.debug(
        "Match score | recipe='%s' | matched=%d/%d | score=%.4f",
        recipe.recipe_name,
        matched,
        len(recipe.ingredients),
        score,
    )
    return score


def _enrich_and_rank_recipes(
    recipes: list[Recipe],
    available_names: frozenset[str],
) -> list[Recipe]:
    """
    Attach ``inventory_match_score`` to each recipe and sort by score descending.

    Args:
        recipes:         Raw Recipe list from the LLM-structured output.
        available_names: Frozenset of active inventory item_names (lowercase).

    Returns:
        List of Recipe objects with ``inventory_match_score`` populated and
        sorted by score descending (most achievable recipe first).
    """
    enriched: list[Recipe] = []
    for recipe in recipes:
        score = _compute_inventory_match_score(recipe, available_names)
        enriched.append(recipe.model_copy(update={"inventory_match_score": score}))

    enriched.sort(key=lambda r: r.inventory_match_score or 0.0, reverse=True)
    return enriched


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

async def recommend_optimized_meals() -> list[Recipe]:
    """
    Generate a prioritized list of Recipe recommendations from the current inventory.

    Full execution pipeline:
      1. Load the live inventory snapshot from ``data/inventory.json``.
      2. Filter out any item with ``count <= 0`` (exhausted / consumed).
      3. Sort the remaining items by ``expiration_date`` ascending so the
         most urgent ingredients appear first in the LLM prompt.
      4. Format and inject the ingredient list into a LangChain chain backed
         by Gemini Pro with ``with_structured_output(_MealRecommendationPayload)``.
      5. Compute ``inventory_match_score`` locally for each returned recipe.
      6. Sort recipes by score descending and return to the caller.

    Returns:
        A list of :class:`~app.models.Recipe` objects, each with:
        - Fully populated ``ingredients``, ``steps``, ``source``.
        - ``restock_recommendations`` for any missing pantry items.
        - ``inventory_match_score`` in [0.0, 1.0] computed against live stock.
        Sorted by ``inventory_match_score`` descending (most achievable first).

        Returns an **empty list** (not an exception) if the inventory is
        completely empty or fully consumed — callers should handle this case
        by prompting the user to restock.

    Raises:
        RuntimeError:
            If the LLM call fails (network error, auth failure, quota exhaustion,
            or an unrecoverable schema validation error) after exhausting retries.
            File-read errors from ``get_current_inventory()`` are propagated as-is.
    """
    # ── Step 1: Load inventory snapshot ──────────────────────────────────────
    logger.info("Recipe advisor: loading inventory snapshot.")
    inventory: InventoryPayload = await get_current_inventory()

    # ── Step 2: Filter exhausted items (count <= 0) ───────────────────────────
    active_items: list[InventoryItem] = [
        it for it in inventory.items if it.count > 0
    ]

    if not active_items:
        logger.warning(
            "Recipe advisor: inventory is empty or fully consumed. "
            "Returning empty recipe list — user should be prompted to restock."
        )
        return []

    logger.info(
        "Recipe advisor: %d active ingredient(s) after filtering (of %d total).",
        len(active_items),
        len(inventory.items),
    )

    # ── Step 3: Sort by expiration_date ascending ─────────────────────────────
    # The sort is defensive — inventory_manager already stores items sorted,
    # but an in-flight mutation or manual file edit could break that invariant.
    active_items.sort(key=lambda it: it.expiration_date)

    # Build lookup structures for Step 5 (score computation)
    available_names: frozenset[str] = frozenset(
        it.item_name.strip().lower() for it in active_items
    )

    # ── Step 4: Format prompt content ────────────────────────────────────────
    ingredient_block = _format_ingredient_block(active_items)
    human_message_text = _build_human_message(ingredient_block)

    logger.debug(
        "Recipe advisor: sending %d ingredients to LLM.\n%s",
        len(active_items),
        ingredient_block,
    )

    # ── Step 4 (cont.): Build LangChain chain with structured output ──────────
    try:
        primary = ChatGoogleGenerativeAI(model=_MODEL_NAME, temperature=_TEMPERATURE, max_retries=_MAX_RETRIES)
        fb1 = ChatGoogleGenerativeAI(model=_FALLBACK_1, temperature=_TEMPERATURE, max_retries=_MAX_RETRIES)
        fb2 = ChatGoogleGenerativeAI(model=_FALLBACK_2, temperature=_TEMPERATURE, max_retries=_MAX_RETRIES)
        
        primary_chain = primary.with_structured_output(_MealRecommendationPayload)
        fb1_chain = fb1.with_structured_output(_MealRecommendationPayload)
        fb2_chain = fb2.with_structured_output(_MealRecommendationPayload)
        
        structured_llm = primary_chain.with_fallbacks([fb1_chain, fb2_chain])

    except Exception as exc:
        logger.error(
            "Recipe advisor: failed to build LangChain chain | error=%s",
            exc,
            exc_info=True,
        )
        raise RuntimeError(
            "Could not initialise the LLM chain for recipe generation. "
            "Verify that GOOGLE_API_KEY is set and langchain_google_genai is installed. "
            f"Underlying error: {exc}"
        ) from exc

    # ── Step 4 (cont.): Invoke LLM ────────────────────────────────────────────
    try:
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=human_message_text),
        ]
        result: _MealRecommendationPayload = await structured_llm.ainvoke(messages)

    except ValueError as exc:
        # Raised by Pydantic when the LLM output violates the Recipe schema.
        logger.error(
            "Recipe advisor: schema validation failed | error=%s",
            exc,
            exc_info=True,
        )
        raise RuntimeError(
            "The LLM returned a recipe payload that failed Pydantic validation. "
            "The inventory context may have been too complex or the model quota was hit. "
            f"Validation error: {exc}"
        ) from exc

    except Exception as exc:
        # Covers: network timeouts, authentication errors, quota exhaustion.
        logger.error(
            "Recipe advisor: unexpected LLM error | error=%s",
            exc,
            exc_info=True,
        )
        raise RuntimeError(
            f"Recipe generation failed due to an unexpected error: {exc}"
        ) from exc

    if not result.recipes:
        logger.warning(
            "Recipe advisor: LLM returned zero recipes despite non-empty inventory. "
            "Inventory context preview: %s",
            ingredient_block[:300],
        )
        return []

    logger.info(
        "Recipe advisor: LLM returned %d recipe(s). Computing match scores.",
        len(result.recipes),
    )

    # ── Step 5: Compute match scores and rank ────────────────────────────────
    ranked_recipes = _enrich_and_rank_recipes(result.recipes, available_names)

    logger.info(
        "Recipe advisor: complete | recipes=%d | top_recipe='%s' | top_score=%.2f",
        len(ranked_recipes),
        ranked_recipes[0].recipe_name if ranked_recipes else "N/A",
        ranked_recipes[0].inventory_match_score if ranked_recipes else 0.0,
    )

    return ranked_recipes


async def precompute_and_cache_recipes() -> None:
    """
    Background worker: generate recommendations and save them to disk.
    This runs asynchronously to avoid blocking the user's upload or consume actions.
    """
    logger.info("Background recipe precomputation started.")
    try:
        recipes = await recommend_optimized_meals()
        
        # Save to cache
        payload = [r.model_dump() for r in recipes]
        
        # Atomic write
        tmp_path = _RECIPE_PATH.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp_path.replace(_RECIPE_PATH)
        
        logger.info("Background recipe precomputation complete. Cached %d recipes.", len(recipes))
    except Exception as exc:
        logger.error("Background recipe precomputation failed | error=%s", exc, exc_info=True)


async def read_cached_recipes() -> list[Recipe]:
    """
    Instantly returns the cached recipes without hitting the LLM.
    """
    if not _RECIPE_PATH.exists():
        return []
        
    try:
        content = _RECIPE_PATH.read_text(encoding="utf-8")
        data = json.loads(content)
        return [Recipe.model_validate(r) for r in data]
    except Exception as exc:
        logger.error("Failed to read cached recipes | error=%s", exc)
        return []
