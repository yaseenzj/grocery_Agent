"""
app/inventory_manager.py
─────────────────────────────────────────────────────────────────────────────
Step 3 of the Grocery Lifecycle Pipeline: Stateful Inventory Mutation.

Owns all reads and writes to data/inventory.json.
Every operation is atomic at the file level: we load → mutate → write in
a single async critical section so concurrent FastAPI requests cannot
produce a torn state.

Two public async functions:

  update_inventory_stock(incoming_stock)
      Merge a freshly enriched InventoryPayload into the persisted store.
      Matching is done on (item_name × expiration_date) so separate purchase
      batches of the same item are tracked independently.

  process_consumption_event(raw_user_speech)
      NL → structured extraction via Gemini → defensive debit of inventory.
      Clamps debits at zero; never produces negative stock.

Pipeline position:
  InventoryPayload → inventory_manager.py → data/inventory.json
  NL speech        → inventory_manager.py → data/inventory.json
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from app.models import InventoryItem, InventoryPayload

# ─────────────────────────────────────────────────────────────────────────────
# Logger & Constants
# ─────────────────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)

_INVENTORY_PATH = Path(__file__).parent.parent / "data" / "inventory.json"
_MODEL_NAME     = "gemini-1.5-pro"
_TEMPERATURE    = 0.0
_MAX_RETRIES    = 2

# Asyncio lock — prevents concurrent coroutines from racing on the JSON file.
# Module-level so it is shared across all requests in the same process.
_FILE_LOCK = asyncio.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# Internal Pydantic Schema for Consumption Extraction
# ─────────────────────────────────────────────────────────────────────────────

class _ConsumptionEvent(BaseModel):
    """
    Transient schema used only within this module.
    Represents a single structured consumption intent extracted from NL speech.
    """

    item_name: str = Field(
        ...,
        description=(
            "Normalized, lowercase grocery item name as it would appear in inventory "
            "(e.g., 'chicken breast', 'whole milk', 'roma tomatoes'). "
            "Expand abbreviations and strip brand names."
        ),
    )
    quantity: float = Field(
        ...,
        gt=0,
        description=(
            "Numeric amount consumed. Must be strictly positive. "
            "Infer from the speech: '1 lb', 'half a carton', '2 pieces' → 1.0, 0.5, 2.0."
        ),
    )
    unit: str = Field(
        ...,
        description=(
            "Unit matching the consumed quantity: 'lbs', 'kg', 'g', 'oz', "
            "'pcs', 'L', 'ml', 'cups', 'tbsp', 'tsp'. "
            "Default to 'pcs' when the speech implies discrete items."
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# File I/O Helpers  (always called inside _FILE_LOCK)
# ─────────────────────────────────────────────────────────────────────────────

async def _load_inventory() -> InventoryPayload:
    """
    Read data/inventory.json from disk and deserialize into InventoryPayload.

    Returns an empty InventoryPayload when:
      - The file does not exist yet (first run).
      - The file exists but is empty or contains only whitespace.
      - The JSON is present but structurally invalid (logged as warning).
    """
    if not _INVENTORY_PATH.exists() or _INVENTORY_PATH.stat().st_size == 0:
        logger.info("Inventory file absent or empty — initializing fresh store.")
        return InventoryPayload(items=[])

    try:
        raw = _INVENTORY_PATH.read_text(encoding="utf-8")
        data = json.loads(raw)
        payload = InventoryPayload.model_validate(data)
        logger.debug("Loaded inventory | items=%d", len(payload.items))
        return payload

    except json.JSONDecodeError as exc:
        logger.warning(
            "inventory.json is malformed JSON — resetting to empty store | error=%s", exc
        )
        return InventoryPayload(items=[])

    except Exception as exc:
        logger.warning(
            "Failed to deserialize inventory.json — resetting to empty store | error=%s", exc
        )
        return InventoryPayload(items=[])


async def _save_inventory(payload: InventoryPayload) -> None:
    """
    Serialize InventoryPayload and atomically write to data/inventory.json.

    Uses a write-to-temp-then-rename strategy to prevent a partial write
    from corrupting the file if the process is interrupted mid-save.

    Args:
        payload: The fully mutated inventory state to persist.
    """
    # Stamp the write time in UTC ISO-8601
    payload.last_updated = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

    _INVENTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = _INVENTORY_PATH.with_suffix(".tmp")
    try:
        tmp_path.write_text(
            payload.model_dump_json(indent=2),
            encoding="utf-8",
        )
        tmp_path.replace(_INVENTORY_PATH)   # Atomic rename on POSIX; best-effort on Windows
        logger.debug("Inventory saved | items=%d | path=%s", len(payload.items), _INVENTORY_PATH)

    except Exception as exc:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        logger.error("Failed to save inventory.json | error=%s", exc, exc_info=True)
        raise RuntimeError(f"Inventory save failed: {exc}") from exc


# ─────────────────────────────────────────────────────────────────────────────
# Merge Helper
# ─────────────────────────────────────────────────────────────────────────────

def _merge_payloads(
    existing: InventoryPayload,
    incoming: InventoryPayload,
) -> InventoryPayload:
    """
    Merge ``incoming`` items into ``existing`` inventory.

    Matching key: ``(item_name, expiration_date)`` — this treats two purchases
    of the same product with different expiration dates as separate batches,
    which is correct for FIFO consumption tracking.

    Behaviour:
      - Match found  → ``count`` is incremented (re-stocking the same batch).
      - No match     → item is appended as a new record.

    Args:
        existing: Current persisted inventory.
        incoming: Freshly enriched payload from expiration.py.

    Returns:
        Merged InventoryPayload sorted by expiration_date ascending.
    """
    # Build a lookup keyed by (item_name, expiration_date) for O(1) matching
    index: dict[tuple[str, str], int] = {
        (it.item_name, it.expiration_date): idx
        for idx, it in enumerate(existing.items)
    }

    merged = list(existing.items)   # Mutable copy

    for new_item in incoming.items:
        key = (new_item.item_name, new_item.expiration_date)
        if key in index:
            pos = index[key]
            old_count = merged[pos].count
            merged[pos] = merged[pos].model_copy(
                update={"count": old_count + new_item.count}
            )
            logger.debug(
                "Restocked | item='%s' | +%.2f %s | new_total=%.2f",
                new_item.item_name,
                new_item.count,
                new_item.unit,
                merged[pos].count,
            )
        else:
            merged.append(new_item)
            index[key] = len(merged) - 1
            logger.debug("New item added | item='%s'", new_item.item_name)

    merged.sort(key=lambda it: it.expiration_date)
    return InventoryPayload(items=merged)


# ─────────────────────────────────────────────────────────────────────────────
# Public API — Function 1
# ─────────────────────────────────────────────────────────────────────────────

async def update_inventory_stock(
    incoming_stock: InventoryPayload,
) -> InventoryPayload:
    """
    Merge freshly purchased/enriched items into the persisted inventory store.

    This is the write path for Step 2 output (expiration.py → here).
    All file operations occur inside ``_FILE_LOCK`` to prevent race conditions
    when multiple requests arrive concurrently.

    Args:
        incoming_stock:
            An :class:`~app.models.InventoryPayload` produced by
            ``expiration.enrich_inventory_lifespans``.
            Must be a non-None InventoryPayload (may have zero items).

    Returns:
        The updated :class:`~app.models.InventoryPayload` as now persisted on disk.

    Raises:
        TypeError:
            If ``incoming_stock`` is not an InventoryPayload instance.
        RuntimeError:
            If the disk write fails after a successful merge.
    """
    if not isinstance(incoming_stock, InventoryPayload):
        raise TypeError(
            f"Expected InventoryPayload, got {type(incoming_stock).__name__}."
        )

    if not incoming_stock.items:
        logger.info("update_inventory_stock called with zero items — no-op.")
        async with _FILE_LOCK:
            return await _load_inventory()

    logger.info(
        "Updating inventory | incoming_items=%d", len(incoming_stock.items)
    )

    async with _FILE_LOCK:
        existing = await _load_inventory()
        merged   = _merge_payloads(existing, incoming_stock)
        await _save_inventory(merged)

    logger.info(
        "Inventory update complete | total_items=%d", len(merged.items)
    )
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# NL Extraction Helper
# ─────────────────────────────────────────────────────────────────────────────

_CONSUMPTION_SYSTEM_PROMPT = """\
You are a precise grocery consumption parser. Your task is to extract \
structured data from a user's natural-language statement about consuming \
grocery items.

Output Rules:
- Return exactly one JSON object matching the schema provided.
- item_name: lowercase, normalized grocery name (strip brands, expand abbreviations).
- quantity: positive float (infer from fractions: "half" → 0.5, "a quarter" → 0.25).
- unit: one of ['lbs', 'kg', 'g', 'oz', 'pcs', 'L', 'ml', 'cups', 'tbsp', 'tsp'].
  Default to 'pcs' for discrete items (eggs, apples, cans).
- Do NOT emit explanations. Return only the JSON object.\
"""


async def _extract_consumption_event(raw_user_speech: str) -> _ConsumptionEvent:
    """
    Use Gemini with structured output to parse a natural-language consumption
    statement into a validated _ConsumptionEvent.

    Args:
        raw_user_speech: Free-form user input, e.g., "I used 1 lb of onions".

    Returns:
        A validated :class:`_ConsumptionEvent`.

    Raises:
        RuntimeError: If the LLM fails or returns an unparseable response.
    """
    try:
        llm = ChatGoogleGenerativeAI(
            model=_MODEL_NAME,
            temperature=_TEMPERATURE,
            max_retries=_MAX_RETRIES,
        )
        structured_llm = llm.with_structured_output(_ConsumptionEvent)

        messages = [
            SystemMessage(content=_CONSUMPTION_SYSTEM_PROMPT),
            HumanMessage(content=raw_user_speech),
        ]

        event: _ConsumptionEvent = await structured_llm.ainvoke(messages)
        logger.info(
            "Consumption extracted | item='%s' | qty=%.2f %s",
            event.item_name,
            event.quantity,
            event.unit,
        )
        return event

    except Exception as exc:
        logger.error(
            "Failed to extract consumption event | speech=%r | error=%s",
            raw_user_speech,
            exc,
            exc_info=True,
        )
        raise RuntimeError(
            f"Could not parse consumption from: {raw_user_speech!r}. "
            f"Underlying error: {exc}"
        ) from exc


# ─────────────────────────────────────────────────────────────────────────────
# Debit Helper
# ─────────────────────────────────────────────────────────────────────────────

def _apply_consumption(
    inventory: InventoryPayload,
    event: _ConsumptionEvent,
) -> tuple[InventoryPayload, bool]:
    """
    Apply a consumption debit to the inventory, enforcing all defensive constraints.

    Matching Strategy — FIFO by expiration_date:
      Items with the same name but different batches are consumed oldest-first
      to minimise waste. The debit cascades across batches until ``event.quantity``
      is fully satisfied or the stock is exhausted.

    Defensive Constraints:
      - Item not in inventory → logs a clear warning; inventory unchanged.
      - Over-consumption      → clamps batch counts to 0.0; never goes negative.
      - Zero-count items      → retained in store (pruned by a later cleanup pass).

    Args:
        inventory: Current inventory state loaded from disk.
        event:     Validated consumption event to apply.

    Returns:
        Tuple of (mutated InventoryPayload, item_was_found: bool).
    """
    target_name = event.item_name.strip().lower()
    remaining_to_consume = event.quantity

    # Find all matching batches, sorted FIFO (earliest expiry first)
    matching_indices: list[int] = [
        idx for idx, it in enumerate(inventory.items)
        if it.item_name == target_name
    ]

    # ── CONSTRAINT 1: Item not found ─────────────────────────────────────────
    if not matching_indices:
        # Attempt fuzzy match (substring) before giving up
        matching_indices = [
            idx for idx, it in enumerate(inventory.items)
            if target_name in it.item_name or it.item_name in target_name
        ]
        if matching_indices:
            matched_name = inventory.items[matching_indices[0]].item_name
            logger.info(
                "Fuzzy match applied | requested='%s' → matched='%s'",
                target_name,
                matched_name,
            )
        else:
            logger.warning(
                "CONSUMPTION REJECTED: Item '%s' not found in inventory. "
                "Current inventory: %s",
                target_name,
                [it.item_name for it in inventory.items],
            )
            return inventory, False   # item_was_found = False

    # Mutable copy
    items = list(inventory.items)

    # ── CONSTRAINT 2: FIFO debit with over-consumption clamping ──────────────
    for idx in matching_indices:
        if remaining_to_consume <= 0:
            break

        batch = items[idx]
        available = batch.count

        if remaining_to_consume >= available:
            # This batch is fully consumed
            deducted = available
            remaining_to_consume -= available
            items[idx] = batch.model_copy(update={"count": 0.0})
            logger.debug(
                "Batch depleted | item='%s' | expiry=%s | consumed=%.2f",
                batch.item_name,
                batch.expiration_date,
                deducted,
            )
        else:
            # Partial debit — batch has remaining stock
            items[idx] = batch.model_copy(
                update={"count": round(available - remaining_to_consume, 6)}
            )
            logger.debug(
                "Partial debit | item='%s' | expiry=%s | consumed=%.2f | remaining=%.2f",
                batch.item_name,
                batch.expiration_date,
                remaining_to_consume,
                items[idx].count,
            )
            remaining_to_consume = 0.0

    if remaining_to_consume > 0:
        logger.warning(
            "OVER-CONSUMPTION DETECTED: '%s' — requested %.2f %s but only %.2f was available. "
            "Stock clamped to 0 across all batches.",
            target_name,
            event.quantity,
            event.unit,
            event.quantity - remaining_to_consume,
        )

    # Preserve sort order
    items.sort(key=lambda it: it.expiration_date)
    return InventoryPayload(items=items), True


# ─────────────────────────────────────────────────────────────────────────────
# Public API — Function 2
# ─────────────────────────────────────────────────────────────────────────────

async def process_consumption_event(raw_user_speech: str) -> InventoryPayload:
    """
    Parse a natural-language consumption statement, debit the inventory,
    and persist the result.

    Full pipeline:
      1. Validate input is non-empty.
      2. Extract structured ``_ConsumptionEvent`` via Gemini structured output.
      3. Acquire ``_FILE_LOCK`` and load inventory from disk.
      4. Apply FIFO debit with defensive clamping.
      5. Write updated inventory back to disk.
      6. Return updated InventoryPayload.

    Args:
        raw_user_speech:
            Free-form natural language describing what was consumed.
            Examples:
            - ``"I used 1 lb of onions"``
            - ``"We finished half the carton of eggs"``
            - ``"Used about 200g of spinach for the salad"``

    Returns:
        The updated :class:`~app.models.InventoryPayload` as now persisted.

    Raises:
        ValueError:
            If ``raw_user_speech`` is empty or whitespace-only.
        RuntimeError:
            If the LLM extraction fails or the disk write fails.
        UserWarning (logged, not raised):
            If the item is not found in inventory — the function returns the
            unchanged inventory rather than raising, keeping the API non-disruptive.
    """
    # ── Guard: reject empty input ─────────────────────────────────────────────
    if not raw_user_speech or not raw_user_speech.strip():
        raise ValueError(
            "raw_user_speech must be a non-empty string describing what was consumed."
        )

    logger.info("Processing consumption event | speech=%r", raw_user_speech)

    # ── Step 1: NL → structured event (outside lock — network call) ───────────
    event = await _extract_consumption_event(raw_user_speech)

    # ── Step 2: Load → debit → save  (inside lock — file operations) ─────────
    async with _FILE_LOCK:
        inventory = await _load_inventory()

        if not inventory.items:
            logger.warning(
                "CONSUMPTION REJECTED: Inventory is empty. "
                "Cannot process: item='%s' qty=%.2f %s",
                event.item_name,
                event.quantity,
                event.unit,
            )
            # Return the empty inventory — do not raise, keep API stable
            return inventory

        updated_inventory, item_found = _apply_consumption(inventory, event)

        if not item_found:
            # Item genuinely absent — return current state without saving
            logger.warning(
                "No changes made to inventory (item '%s' not found).",
                event.item_name,
            )
            return inventory

        await _save_inventory(updated_inventory)

    logger.info(
        "Consumption event persisted | item='%s' | debited=%.2f %s | "
        "remaining_batches=%d",
        event.item_name,
        event.quantity,
        event.unit,
        sum(
            1 for it in updated_inventory.items
            if it.item_name == event.item_name and it.count > 0
        ),
    )

    return updated_inventory


# ─────────────────────────────────────────────────────────────────────────────
# Utility: Read-Only Snapshot
# ─────────────────────────────────────────────────────────────────────────────

async def get_current_inventory() -> InventoryPayload:
    """
    Return a read-only snapshot of the current inventory.

    Used by FastAPI GET endpoints and recipe_advisor.py to inspect state
    without triggering a write. Still acquires the lock to ensure a
    consistent read if a concurrent write is in progress.

    Returns:
        The current :class:`~app.models.InventoryPayload` from disk.
    """
    async with _FILE_LOCK:
        return await _load_inventory()
