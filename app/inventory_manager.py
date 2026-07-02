from __future__ import annotations
import asyncio
import logging
import re
from datetime import datetime
from typing import Literal, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from app.database import get_db_pool
from app.models import InventoryItem, InventoryPayload
logger = logging.getLogger(__name__)
_MODEL_NAME = 'gemini-3.5-flash'
_FALLBACK_1 = 'gemini-2.5-flash'
_FALLBACK_2 = 'gemini-1.5-flash'
_TEMPERATURE = 0.0
_MAX_RETRIES = 5
_DB_LOCK = asyncio.Lock()

def _validate_input_guardrail(text: str) -> None:
    if not text or not text.strip():
        raise ValueError('Input statement cannot be empty.')
    if len(text) > 150:
        raise ValueError('Input statement is too long (maximum 150 characters).')
    injection_keywords = ['ignore instructions', 'system prompt', 'bypass security', 'delete database', 'drop table']
    text_lower = text.lower()
    for keyword in injection_keywords:
        if keyword in text_lower:
            raise ValueError('Potential injection attack detected in query.')
    if re.search('[\\";\\-\\-]', text):
        raise ValueError('Invalid characters detected in query.')
AllowedUnits = Literal['lbs', 'kg', 'g', 'oz', 'pcs', 'L', 'ml', 'cups', 'tbsp', 'tsp']

class _ConsumptionEvent(BaseModel):
    item_name: str = Field(..., description="Normalized, lowercase grocery item name as it would appear in inventory (e.g., 'chicken breast', 'whole milk', 'roma tomatoes'). Expand abbreviations and strip brand names.")
    quantity: float = Field(..., gt=0, description="Numeric amount consumed. Must be strictly positive. Infer from the speech: '1 lb', 'half a carton', '2 pieces' → 1.0, 0.5, 2.0.")
    unit: AllowedUnits = Field(..., description='Unit matching the consumed quantity.')

async def _load_inventory() -> InventoryPayload:
    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch('SELECT item_name, count, unit, purchase_date, expiration_date FROM inventory_items ORDER BY expiration_date ASC')
            items = []
            for r in rows:
                items.append(InventoryItem(item_name=r['item_name'], count=r['count'], unit=r['unit'], purchase_date=r['purchase_date'].isoformat(), expiration_date=r['expiration_date'].isoformat()))
            last_updated_row = await conn.fetchrow('SELECT MAX(updated_at) as last_updated FROM inventory_items')
            last_updated = None
            if last_updated_row and last_updated_row['last_updated']:
                last_updated = last_updated_row['last_updated'].strftime('%Y-%m-%dT%H:%M:%S')
            return InventoryPayload(items=items, last_updated=last_updated)
    except Exception as exc:
        logger.error('Failed to load inventory from PostgreSQL | error=%s', exc, exc_info=True)
        raise RuntimeError(f'Database read failed: {exc}') from exc

async def _save_inventory(payload: InventoryPayload) -> None:
    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute('DELETE FROM inventory_items')
                if payload.items:
                    values = []
                    for item in payload.items:
                        p_date = datetime.strptime(item.purchase_date, '%Y-%m-%d').date()
                        e_date = datetime.strptime(item.expiration_date, '%Y-%m-%d').date()
                        values.append((item.item_name, item.count, item.unit, p_date, e_date))
                    await conn.executemany('INSERT INTO inventory_items (item_name, count, unit, purchase_date, expiration_date) VALUES ($1, $2, $3, $4, $5)', values)
        logger.debug('Inventory successfully saved to PostgreSQL | items=%d', len(payload.items))
    except Exception as exc:
        logger.error('Failed to save inventory to PostgreSQL | error=%s', exc, exc_info=True)
        raise RuntimeError(f'Database write failed: {exc}') from exc

def _merge_payloads(existing: InventoryPayload, incoming: InventoryPayload) -> InventoryPayload:
    index: dict[tuple[str, str], int] = {(it.item_name, it.expiration_date): idx for idx, it in enumerate(existing.items)}
    merged = list(existing.items)
    for new_item in incoming.items:
        key = (new_item.item_name, new_item.expiration_date)
        if key in index:
            pos = index[key]
            old_count = merged[pos].count
            merged[pos] = merged[pos].model_copy(update={'count': old_count + new_item.count})
            logger.debug("Restocked | item='%s' | +%.2f %s | new_total=%.2f", new_item.item_name, new_item.count, new_item.unit, merged[pos].count)
        else:
            merged.append(new_item)
            index[key] = len(merged) - 1
            logger.debug("New item added | item='%s'", new_item.item_name)
    merged.sort(key=lambda it: it.expiration_date)
    return InventoryPayload(items=merged)

async def update_inventory_stock(incoming_stock: InventoryPayload) -> InventoryPayload:
    if not isinstance(incoming_stock, InventoryPayload):
        raise TypeError(f'Expected InventoryPayload, got {type(incoming_stock).__name__}.')
    if not incoming_stock.items:
        logger.info('update_inventory_stock called with zero items — no-op.')
        async with _DB_LOCK:
            return await _load_inventory()
    logger.info('Updating inventory in DB | incoming_items=%d', len(incoming_stock.items))
    async with _DB_LOCK:
        existing = await _load_inventory()
        merged = _merge_payloads(existing, incoming_stock)
        await _save_inventory(merged)
    logger.info('Inventory update complete | total_items=%d', len(merged.items))
    return merged
_CONSUMPTION_SYSTEM_PROMPT = 'You are a precise grocery consumption parser. Your task is to extract structured data from a user\'s natural-language statement about consuming grocery items.\n\nOutput Rules:\n- Return exactly one JSON object matching the schema provided.\n- item_name: lowercase, normalized grocery name (strip brands, expand abbreviations).\n- quantity: positive float (infer from fractions: "half" → 0.5, "a quarter" → 0.25).\n- unit: one of [\'lbs\', \'kg\', \'g\', \'oz\', \'pcs\', \'L\', \'ml\', \'cups\', \'tbsp\', \'tsp\'].\n  Default to \'pcs\' for discrete items (eggs, apples, cans).\n- Do NOT emit explanations. Return only the JSON object.'

async def _extract_consumption_event(raw_user_speech: str) -> _ConsumptionEvent:
    try:
        primary = ChatGoogleGenerativeAI(model=_MODEL_NAME, temperature=_TEMPERATURE, max_retries=_MAX_RETRIES)
        fb1 = ChatGoogleGenerativeAI(model=_FALLBACK_1, temperature=_TEMPERATURE, max_retries=_MAX_RETRIES)
        fb2 = ChatGoogleGenerativeAI(model=_FALLBACK_2, temperature=_TEMPERATURE, max_retries=_MAX_RETRIES)
        primary_chain = primary.with_structured_output(_ConsumptionEvent)
        fb1_chain = fb1.with_structured_output(_ConsumptionEvent)
        fb2_chain = fb2.with_structured_output(_ConsumptionEvent)
        structured_llm = primary_chain.with_fallbacks([fb1_chain, fb2_chain])
        messages = [SystemMessage(content=_CONSUMPTION_SYSTEM_PROMPT), HumanMessage(content=raw_user_speech)]
        event: _ConsumptionEvent = await structured_llm.ainvoke(messages)
        logger.info("Consumption extracted | item='%s' | qty=%.2f %s", event.item_name, event.quantity, event.unit)
        return event
    except Exception as exc:
        logger.error('Failed to extract consumption event | speech=%r | error=%s', raw_user_speech, exc, exc_info=True)
        raise RuntimeError(f'Could not parse consumption from: {raw_user_speech!r}. Underlying error: {exc}') from exc

def _apply_consumption(inventory: InventoryPayload, event: _ConsumptionEvent) -> tuple[InventoryPayload, bool]:
    target_name = event.item_name.strip().lower()
    remaining_to_consume = event.quantity
    matching_indices: list[int] = [idx for idx, it in enumerate(inventory.items) if it.item_name == target_name]
    if not matching_indices:
        matching_indices = [idx for idx, it in enumerate(inventory.items) if target_name in it.item_name or it.item_name in target_name]
        if matching_indices:
            matched_name = inventory.items[matching_indices[0]].item_name
            logger.info("Fuzzy match applied | requested='%s' → matched='%s'", target_name, matched_name)
        else:
            logger.warning("CONSUMPTION REJECTED: Item '%s' not found in inventory. Current inventory: %s", target_name, [it.item_name for it in inventory.items])
            return (inventory, False)
    items = list(inventory.items)
    for idx in matching_indices:
        if remaining_to_consume <= 0:
            break
        batch = items[idx]
        available = batch.count
        if remaining_to_consume >= available:
            deducted = available
            remaining_to_consume -= available
            items[idx] = batch.model_copy(update={'count': 0.0})
            logger.debug("Batch depleted | item='%s' | expiry=%s | consumed=%.2f", batch.item_name, batch.expiration_date, deducted)
        else:
            items[idx] = batch.model_copy(update={'count': round(available - remaining_to_consume, 6)})
            logger.debug("Partial debit | item='%s' | expiry=%s | consumed=%.2f | remaining=%.2f", batch.item_name, batch.expiration_date, remaining_to_consume, items[idx].count)
            remaining_to_consume = 0.0
    if remaining_to_consume > 0:
        logger.warning("OVER-CONSUMPTION DETECTED: '%s' — requested %.2f %s but only %.2f was available. Stock clamped to 0 across all batches.", target_name, event.quantity, event.unit, event.quantity - remaining_to_consume)
    items.sort(key=lambda it: it.expiration_date)
    return (InventoryPayload(items=items), True)

async def process_consumption_event(raw_user_speech: str) -> InventoryPayload:
    _validate_input_guardrail(raw_user_speech)
    logger.info('Processing consumption event | speech=%r', raw_user_speech)
    event = await _extract_consumption_event(raw_user_speech)
    async with _DB_LOCK:
        inventory = await _load_inventory()
        if not inventory.items:
            logger.warning("CONSUMPTION REJECTED: Inventory is empty. Cannot process: item='%s' qty=%.2f %s", event.item_name, event.quantity, event.unit)
            return inventory
        updated_inventory, item_found = _apply_consumption(inventory, event)
        if not item_found:
            logger.warning("No changes made to inventory (item '%s' not found).", event.item_name)
            return inventory
        await _save_inventory(updated_inventory)
    logger.info("Consumption event persisted in DB | item='%s' | debited=%.2f %s | remaining_batches=%d", event.item_name, event.quantity, event.unit, sum((1 for it in updated_inventory.items if it.item_name == event.item_name and it.count > 0)))
    return updated_inventory

async def get_current_inventory() -> InventoryPayload:
    async with _DB_LOCK:
        return await _load_inventory()

async def remove_inventory_item(item_name: str) -> InventoryPayload:
    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            async with _DB_LOCK:
                await conn.execute('DELETE FROM inventory_items WHERE LOWER(item_name) = $1', item_name.lower())
                logger.info("Purged all batches of item '%s' from DB", item_name)
                return await _load_inventory()
    except Exception as exc:
        logger.error("Failed to remove item '%s' from DB | error=%s", item_name, exc, exc_info=True)
        raise RuntimeError(f'Database deletion failed: {exc}') from exc