"""
app/expiration.py
─────────────────────────────────────────────────────────────────────────────
Step 2 of the Grocery Lifecycle Pipeline: Expiration Date Calculation Layer.

Uses a hybrid cost-optimization strategy:
  1. LOCAL LOOKUP  — O(1) dict hit against a curated shelf-life knowledge base.
  2. LLM FALLBACK  — Atomic Gemini call only for items absent from the local DB.
  3. SAFE DEFAULT  — 7-day fallback when even the LLM returns an unusable value.

The result is a fully enriched InventoryPayload where every item carries a
computed expiration_date, ready for inventory_manager.py to persist.

Pipeline position:
  ReceiptPayload → expiration.py → InventoryPayload → inventory_manager.py → ...
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, timedelta
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI

from app.models import InventoryItem, InventoryPayload, ReceiptItem, ReceiptPayload

# ─────────────────────────────────────────────────────────────────────────────
# Logger
# ─────────────────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# LLM Configuration (Fallback Only)
# ─────────────────────────────────────────────────────────────────────────────

_MODEL_NAME  = "gemini-1.5-pro"
_TEMPERATURE = 0.0   # Must be deterministic — we parse the response as int
_MAX_RETRIES = 2
_DEFAULT_SHELF_LIFE_DAYS = 7   # Conservative safe-default when all lookups fail

# ─────────────────────────────────────────────────────────────────────────────
# Local Shelf-Life Knowledge Base  (refrigerated storage, days)
# ─────────────────────────────────────────────────────────────────────────────
# Keys are lowercase, single-word or short-phrase names that mirror the
# normalized item_name values produced by parser.py.
# Organized by category for maintainability; lookup is O(1) via dict.
# ─────────────────────────────────────────────────────────────────────────────

SHELF_LIFE_DB: dict[str, int] = {
    # ── Dairy & Eggs ─────────────────────────────────────────────────────────
    "milk":                   7,
    "whole milk":             7,
    "skim milk":              7,
    "buttermilk":             7,
    "heavy cream":            10,
    "half and half":          10,
    "sour cream":             14,
    "cream cheese":           14,
    "cottage cheese":         10,
    "ricotta":                7,
    "butter":                 30,
    "ghee":                   90,
    "yogurt":                 14,
    "greek yogurt":           17,
    "eggs":                   35,
    "hard boiled eggs":       7,
    # ── Soft Cheeses ─────────────────────────────────────────────────────────
    "mozzarella":             7,
    "fresh mozzarella":       5,
    "brie":                   14,
    "camembert":              14,
    "feta":                   28,
    "goat cheese":            14,
    "ricotta salata":         30,
    # ── Hard / Semi-Hard Cheeses ─────────────────────────────────────────────
    "cheddar":                60,
    "parmesan":               90,
    "parmigiano reggiano":    90,
    "gouda":                  60,
    "swiss cheese":           60,
    "gruyere":                60,
    "manchego":               60,
    "provolone":              45,
    "colby":                  60,
    "pepper jack":            60,
    # ── Meat — Raw ───────────────────────────────────────────────────────────
    "chicken breast":         2,
    "chicken thighs":         2,
    "whole chicken":          2,
    "ground beef":            2,
    "beef steak":             3,
    "beef roast":             5,
    "pork chops":             3,
    "pork loin":              5,
    "ground pork":            2,
    "lamb chops":             3,
    "ground lamb":            2,
    "veal":                   3,
    "turkey":                 2,
    "ground turkey":          2,
    "duck":                   2,
    "bacon":                  7,
    "ham":                    5,
    "prosciutto":             5,
    "salami":                 14,
    "pepperoni":              14,
    "hot dogs":               14,
    "sausage":                5,
    # ── Seafood — Raw ────────────────────────────────────────────────────────
    "salmon":                 2,
    "tilapia":                2,
    "tuna steak":             2,
    "cod":                    2,
    "halibut":                2,
    "shrimp":                 2,
    "scallops":               2,
    "clams":                  2,
    "mussels":                2,
    "oysters":                5,
    "crab":                   2,
    "lobster":                2,
    "canned tuna":            365,   # Unopened
    # ── Cooked / Deli ────────────────────────────────────────────────────────
    "cooked chicken":         4,
    "cooked beef":            4,
    "deli turkey":            5,
    "deli ham":               5,
    "rotisserie chicken":     4,
    "leftovers":              4,
    # ── Vegetables — Leafy ───────────────────────────────────────────────────
    "spinach":                5,
    "baby spinach":           5,
    "arugula":                5,
    "romaine lettuce":        7,
    "iceberg lettuce":        10,
    "kale":                   7,
    "swiss chard":            5,
    "collard greens":         7,
    "cabbage":                14,
    "red cabbage":            14,
    "bok choy":               5,
    "brussels sprouts":       7,
    # ── Vegetables — Cruciferous & Stem ──────────────────────────────────────
    "broccoli":               5,
    "cauliflower":            7,
    "asparagus":              4,
    "celery":                 14,
    "fennel":                 7,
    "artichoke":              7,
    # ── Vegetables — Fruiting ────────────────────────────────────────────────
    "bell pepper":            10,
    "red bell pepper":        10,
    "green bell pepper":      10,
    "jalapeno":               14,
    "serrano pepper":         14,
    "zucchini":               7,
    "yellow squash":          7,
    "eggplant":               5,
    "cucumber":               7,
    "tomato":                 7,
    "roma tomatoes":          7,
    "cherry tomatoes":        10,
    "grape tomatoes":         10,
    "corn":                   3,
    "okra":                   4,
    # ── Vegetables — Root & Tuber ────────────────────────────────────────────
    "carrots":                21,
    "baby carrots":           21,
    "potatoes":               21,
    "sweet potatoes":         21,
    "beets":                  14,
    "parsnips":               14,
    "turnips":                14,
    "radishes":               14,
    "ginger":                 21,
    "garlic":                 30,
    "onion":                  30,
    "red onion":              30,
    "shallots":               30,
    "leeks":                  14,
    "scallions":              7,
    "green onions":           7,
    # ── Vegetables — Fungal ──────────────────────────────────────────────────
    "mushrooms":              7,
    "cremini mushrooms":      7,
    "portobello mushrooms":   7,
    "shiitake mushrooms":     10,
    # ── Fruits — Citrus ──────────────────────────────────────────────────────
    "lemons":                 21,
    "limes":                  21,
    "oranges":                21,
    "grapefruit":             21,
    "clementines":            14,
    "mandarins":              14,
    # ── Fruits — Berries ─────────────────────────────────────────────────────
    "strawberries":           5,
    "blueberries":            10,
    "raspberries":            3,
    "blackberries":           5,
    "cranberries":            28,
    # ── Fruits — Tropical ────────────────────────────────────────────────────
    "bananas":                5,
    "mangoes":                5,
    "pineapple":              5,
    "papaya":                 5,
    "kiwi":                   14,
    "avocado":                4,
    "coconut":                14,
    # ── Fruits — Stone ───────────────────────────────────────────────────────
    "peaches":                5,
    "nectarines":             5,
    "plums":                  5,
    "cherries":               7,
    "apricots":               5,
    # ── Fruits — Pome & Other ────────────────────────────────────────────────
    "apples":                 42,
    "pears":                  7,
    "grapes":                 10,
    "watermelon":             10,
    "cantaloupe":             5,
    "honeydew":               7,
    # ── Bread & Bakery ───────────────────────────────────────────────────────
    "bread":                  7,
    "sourdough bread":        7,
    "whole wheat bread":      7,
    "bagels":                 7,
    "english muffins":        7,
    "tortillas":              14,
    "flour tortillas":        14,
    "corn tortillas":         14,
    "pita bread":             7,
    "baguette":               3,
    "croissants":             3,
    "muffins":                5,
    # ── Condiments & Sauces ──────────────────────────────────────────────────
    "ketchup":                180,
    "mustard":                180,
    "mayonnaise":             90,
    "hot sauce":              180,
    "soy sauce":              365,
    "worcestershire sauce":   365,
    "salsa":                  14,
    "hummus":                 10,
    "guacamole":              3,
    "pesto":                  7,
    "tomato sauce":           5,     # Opened jar
    "pasta sauce":            5,
    "ranch dressing":         30,
    "italian dressing":       60,
    "balsamic vinegar":       365,
    # ── Beverages ────────────────────────────────────────────────────────────
    "orange juice":           7,
    "apple juice":            7,
    "coconut water":          5,
    "almond milk":            7,
    "oat milk":               7,
    "soy milk":               7,
    # ── Pantry / Dry Goods (common fridge items) ─────────────────────────────
    "tofu":                   5,
    "tempeh":                 10,
    "cooked rice":            5,
    "cooked pasta":           5,
    "cooked beans":           5,
    "open canned goods":      4,
    "olives":                 30,
    "capers":                 30,
    "pickles":                90,
    "kimchi":                 90,
    "sauerkraut":             60,
    "miso paste":             365,
    # ── Herbs — Fresh ────────────────────────────────────────────────────────
    "basil":                  7,
    "cilantro":               7,
    "parsley":                10,
    "mint":                   10,
    "dill":                   10,
    "thyme":                  14,
    "rosemary":               14,
    "chives":                 10,
    "sage":                   14,
    "tarragon":               10,
    # ── Nuts & Seeds (fridge-stored) ─────────────────────────────────────────
    "walnuts":                180,
    "almonds":                180,
    "cashews":                180,
    "pecans":                 180,
    "pine nuts":              90,
    "sesame seeds":           180,
    "flaxseeds":              90,
}

# ─────────────────────────────────────────────────────────────────────────────
# Local Lookup Helper
# ─────────────────────────────────────────────────────────────────────────────

def _local_shelf_life(item_name: str) -> Optional[int]:
    """
    O(1) dictionary lookup against the local shelf-life knowledge base.

    Performs two attempts:
      1. Exact match on the normalized item_name.
      2. Partial / substring match — catches cases where parser.py produces
         a slightly more verbose name (e.g., "organic baby spinach" → "baby spinach").

    Args:
        item_name: Normalized, lowercase item name from a ReceiptItem.

    Returns:
        Shelf-life in days if found, else None.
    """
    # Exact match (fast path)
    if item_name in SHELF_LIFE_DB:
        return SHELF_LIFE_DB[item_name]

    # Substring match: check if any known key is contained within item_name
    # (e.g., item_name="organic roma tomatoes" → key="roma tomatoes" → hit)
    for key, days in SHELF_LIFE_DB.items():
        if key in item_name or item_name in key:
            logger.debug("Partial shelf-life match: '%s' → '%s' (%d days)", item_name, key, days)
            return days

    return None


# ─────────────────────────────────────────────────────────────────────────────
# LLM Fallback Helper
# ─────────────────────────────────────────────────────────────────────────────

async def _llm_shelf_life(item_name: str) -> int:
    """
    Atomic fallback: ask Gemini for the refrigerated shelf-life of an unknown item.

    Uses the most terse possible prompt to minimise token cost and avoid
    the LLM adding prose around the numeric answer.

    Args:
        item_name: Item not found in the local knowledge base.

    Returns:
        Parsed integer shelf-life in days.
        Falls back to ``_DEFAULT_SHELF_LIFE_DAYS`` if the response is
        non-numeric or the API call fails.
    """
    prompt = (
        f"Provide only an integer representing the typical refrigerated shelf life "
        f"in days for {item_name}. "
        f"Do not reply with text or units, return only the raw number."
    )

    logger.info("LLM fallback shelf-life lookup | item='%s'", item_name)

    try:
        llm = ChatGoogleGenerativeAI(
            model=_MODEL_NAME,
            temperature=_TEMPERATURE,
            max_retries=_MAX_RETRIES,
        )
        response = await llm.ainvoke(prompt)
        raw: str = response.content.strip()

        # Parse — strip any accidental unit suffix the model may include
        # (e.g., "7 days" → "7", "~10" → "10")
        numeric_part = "".join(ch for ch in raw.split()[0] if ch.isdigit())
        if not numeric_part:
            raise ValueError(f"Non-numeric LLM response: {raw!r}")

        days = int(numeric_part)
        if days <= 0:
            raise ValueError(f"LLM returned non-positive shelf life: {days}")

        logger.info("LLM shelf-life resolved | item='%s' → %d days", item_name, days)
        return days

    except Exception as exc:
        logger.warning(
            "LLM shelf-life lookup failed for '%s' — using default %d days | error=%s",
            item_name,
            _DEFAULT_SHELF_LIFE_DAYS,
            exc,
        )
        return _DEFAULT_SHELF_LIFE_DAYS


# ─────────────────────────────────────────────────────────────────────────────
# Per-Item Enrichment
# ─────────────────────────────────────────────────────────────────────────────

async def _enrich_item(item: ReceiptItem) -> InventoryItem:
    """
    Resolve the shelf-life for a single ReceiptItem and produce an InventoryItem.

    Resolution order:
      1. Local DB hit  → zero network cost
      2. LLM fallback  → one atomic Gemini call
      3. Safe default  → _DEFAULT_SHELF_LIFE_DAYS if both fail

    Args:
        item: A validated ReceiptItem from the parsed receipt.

    Returns:
        InventoryItem with expiration_date populated.
    """
    # ── Step 1: Local lookup ──────────────────────────────────────────────────
    shelf_life: Optional[int] = _local_shelf_life(item.item_name)
    source = "local_db"

    # ── Step 2: LLM fallback ─────────────────────────────────────────────────
    if shelf_life is None:
        shelf_life = await _llm_shelf_life(item.item_name)
        source = "llm_fallback"

    logger.debug(
        "Shelf life resolved | item='%s' | days=%d | source=%s",
        item.item_name,
        shelf_life,
        source,
    )

    # ── Step 3: Compute expiration_date ──────────────────────────────────────
    try:
        purchase = date.fromisoformat(item.purchase_date)
    except ValueError as exc:
        logger.warning(
            "Invalid purchase_date '%s' for item '%s' — using today | error=%s",
            item.purchase_date,
            item.item_name,
            exc,
        )
        purchase = date.today()

    expiration = purchase + timedelta(days=shelf_life)

    return InventoryItem(
        item_name=item.item_name,
        count=item.count,
        unit=item.unit,
        purchase_date=item.purchase_date,
        expiration_date=expiration.isoformat(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

async def enrich_inventory_lifespans(receipt: ReceiptPayload) -> InventoryPayload:
    """
    Enrich all items in a parsed receipt with computed expiration dates.

    Processes all items **concurrently** via ``asyncio.gather`` so that
    multiple LLM fallback calls (for unknown items) execute in parallel
    rather than sequentially — critical for large receipts.

    Items that fully resolve from the local DB incur zero network cost and
    complete synchronously within the gather, keeping latency minimal.

    The returned InventoryPayload is sorted by ``expiration_date`` ascending
    so that inventory_manager.py and recipe_advisor.py always operate on
    the most-urgent items first.

    Args:
        receipt:
            A validated ReceiptPayload produced by ``parser.parse_receipt_text``.
            May contain zero items (returns an empty InventoryPayload).

    Returns:
        A fully enriched :class:`~app.models.InventoryPayload` with
        ``last_updated`` set to today's ISO date.

    Raises:
        TypeError:
            If ``receipt`` is not a ReceiptPayload instance.
        RuntimeError:
            If a critical, unrecoverable error occurs during enrichment.
            Individual item failures are handled gracefully with fallbacks
            and will never raise here.
    """
    if not isinstance(receipt, ReceiptPayload):
        raise TypeError(
            f"Expected a ReceiptPayload instance, got {type(receipt).__name__}. "
            "Ensure parser.parse_receipt_text() completed successfully before calling this function."
        )

    if not receipt.items:
        logger.warning(
            "enrich_inventory_lifespans called with an empty ReceiptPayload. "
            "Returning empty InventoryPayload."
        )
        return InventoryPayload(items=[], last_updated=date.today().isoformat())

    logger.info(
        "Starting expiration enrichment | items=%d | purchase_date=%s",
        len(receipt.items),
        receipt.date_of_purchase,
    )

    # ── Concurrent enrichment ─────────────────────────────────────────────────
    # asyncio.gather runs all _enrich_item coroutines concurrently.
    # return_exceptions=True ensures one failing item never cancels the batch.
    results = await asyncio.gather(
        *[_enrich_item(item) for item in receipt.items],
        return_exceptions=True,
    )

    # ── Collect successful results, log failures ──────────────────────────────
    inventory_items: list[InventoryItem] = []
    for idx, result in enumerate(results):
        if isinstance(result, Exception):
            failed_name = receipt.items[idx].item_name
            logger.error(
                "Enrichment failed for item '%s' (index %d) — skipping | error=%s",
                failed_name,
                idx,
                result,
            )
        else:
            inventory_items.append(result)

    # ── Sort by expiration_date ascending (most urgent first) ─────────────────
    inventory_items.sort(key=lambda it: it.expiration_date)

    logger.info(
        "Expiration enrichment complete | enriched=%d | failed=%d",
        len(inventory_items),
        len(receipt.items) - len(inventory_items),
    )

    return InventoryPayload(
        items=inventory_items,
        last_updated=date.today().isoformat(),
    )
