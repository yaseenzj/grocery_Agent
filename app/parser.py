"""
app/parser.py
─────────────────────────────────────────────────────────────────────────────
Step 1 of the Grocery Lifecycle Pipeline: Structured Receipt Ingestion.

Converts raw, unstructured receipt text (OCR output, typed notes, markdown)
into a fully validated ReceiptPayload using LangChain's .with_structured_output()
binding against Gemini Pro.

The LLM acts as a data-extraction compiler — it never generates free-form prose,
only deterministic JSON that Pydantic v2 validates on arrival.

Pipeline position:
  [Raw Text] → parser.py → ReceiptPayload → expiration.py → ...
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from app.models import ReceiptPayload

# ─────────────────────────────────────────────────────────────────────────────
# Logger
# ─────────────────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# LLM Configuration
# ─────────────────────────────────────────────────────────────────────────────

_MODEL_NAME = "gemini-3.5-flash"  
_FALLBACK_1 = "gemini-2.5-flash"
_FALLBACK_2 = "gemini-1.5-flash"
_TEMPERATURE = 0.0              
_MAX_RETRIES = 5                

def _build_robust_structured_llm(schema) -> Any:
    """
    Creates a highly available structured LLM chain.
    If the primary model (gemini-3.5-flash) fails due to a 503 Overload, 
    it automatically cascades down to older models to ensure 100% uptime.
    """
    primary = ChatGoogleGenerativeAI(model=_MODEL_NAME, temperature=_TEMPERATURE, max_retries=_MAX_RETRIES)
    fb1 = ChatGoogleGenerativeAI(model=_FALLBACK_1, temperature=_TEMPERATURE, max_retries=_MAX_RETRIES)
    fb2 = ChatGoogleGenerativeAI(model=_FALLBACK_2, temperature=_TEMPERATURE, max_retries=_MAX_RETRIES)
    
    primary_chain = primary.with_structured_output(schema)
    fb1_chain = fb1.with_structured_output(schema)
    fb2_chain = fb2.with_structured_output(schema)
    
    return primary_chain.with_fallbacks([fb1_chain, fb2_chain])


# ─────────────────────────────────────────────────────────────────────────────
# Prompt Template
# ─────────────────────────────────────────────────────────────────────────────

# The system prompt is parameterized with {today_date} so the LLM always has
# a concrete fallback date without hard-coding any value at import time.
_SYSTEM_PROMPT_TEMPLATE = """\
You are an isolated data-extraction compiler. Your sole objective is to \
convert ANY form of grocery input — whether a formal store receipt, \
a casual comma-separated list, bullet points, or freeform text — into a \
standardized JSON structure matching the required schema.

Accepted input formats (non-exhaustive):
- Formal store receipts: "Eggplant 2 lbs\nPotatoes 1 lbs"
- Casual comma lists: "1 litre milk, 10 tomatoes, 3 eggs"
- Bullet lists: "- 2 lbs chicken\n- 1 gallon milk"
- Freeform notes: "bought some spinach and 2 cartons of eggs"

Operational Directives:

1. Normalize item descriptions: Clean raw or noisy text into \
clean, descriptive, human-readable item names \
(e.g., change "BH Fresh Mozzarella Ball" into "mozzarella ball"). \
All item_name values must be lowercase.

2. Quantity Coercion: Extract fractional weights or integers into the \
appropriate `count` fields. Ensure values match common units: \
'lbs', 'kg', 'g', 'oz', 'pcs', 'L', 'ml', 'litre', 'gallon'. \
When the unit is ambiguous, infer from context \
(e.g., produce sold by weight → 'lbs'; packaged goods → 'pcs'). \
If no quantity is given, default count to 1.

3. Temporal Mapping: Isolate the purchase date if printed on the receipt. \
Format it as ISO-8601 (YYYY-MM-DD). \
If no explicit date is present, default to the context-provided baseline: {today_date}. \
Apply this same date to every ReceiptItem.purchase_date and to date_of_purchase.

4. Deduplication: If the same item appears on multiple lines \
(e.g., bought twice), merge them into a single ReceiptItem by summing `count`.

5. Exclusions: Ignore non-product lines such as subtotals, taxes, \
loyalty points, cashier names, store addresses, and payment method rows.

6. CRITICAL: You MUST always attempt to extract items. Even a single item \
should be returned in the items list. NEVER return an empty items list if \
any food or grocery item is mentioned in the input.

Output ONLY the structured JSON payload. Do not emit explanations, \
preambles, or markdown fences.\
"""

_HUMAN_PROMPT_TEMPLATE = """\
Parse the following grocery receipt and return a ReceiptPayload JSON object.

--- RECEIPT START ---
{receipt_text}
--- RECEIPT END ---\
"""

_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM_PROMPT_TEMPLATE),
        ("human", _HUMAN_PROMPT_TEMPLATE),
    ]
)


# ─────────────────────────────────────────────────────────────────────────────
# Structured Chain Builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_chain() -> Any:
    """
    Compose the LangChain LCEL chain:

        prompt  →  llm.with_structured_output(ReceiptPayload)

    .with_structured_output() binds the Pydantic schema as a function-calling
    tool, forcing Gemini Pro to emit JSON that satisfies ReceiptPayload's field
    constraints before the response reaches Python — no post-hoc parsing needed.
    """
    structured_llm = _build_robust_structured_llm(ReceiptPayload)

    # LCEL pipe: format the prompt messages first, then invoke structured LLM
    chain = _PROMPT | structured_llm
    return chain


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

async def parse_receipt_text(
    receipt_text: str,
    today_date: str,
) -> ReceiptPayload:
    """
    Parse raw receipt text into a validated ReceiptPayload.

    This is the primary entry point for Step 1 of the grocery lifecycle pipeline.
    It is intentionally stateless — no file I/O or side effects occur here.
    The caller (main.py route handler) is responsible for persisting the result.

    Args:
        receipt_text:
            Raw, unstructured text from the receipt. Acceptable formats include:
            - Plain text (OCR output, hand-typed notes)
            - Markdown tables or bullet lists
            - Mixed noisy strings with extra whitespace / symbols

        today_date:
            Fallback purchase date in ISO-8601 format (YYYY-MM-DD).
            Injected into the system prompt so the LLM can assign a concrete
            date when the receipt omits one. Callers should pass
            ``datetime.date.today().isoformat()`` or the equivalent.

    Returns:
        A fully validated :class:`~app.models.ReceiptPayload` instance.
        All ReceiptItem fields are guaranteed to conform to Pydantic v2
        constraints at the time of return.

    Raises:
        ValueError:
            If ``receipt_text`` is empty or contains only whitespace.
        RuntimeError:
            If the LLM returns a response that cannot be coerced into a
            valid ReceiptPayload after exhausting retries.
        Exception:
            Propagated as-is for unexpected upstream errors (e.g., network
            timeouts, authentication failures) after logging.
    """
    # ── Guard: reject obviously empty inputs early ────────────────────────────
    if not receipt_text or not receipt_text.strip():
        raise ValueError(
            "receipt_text must be a non-empty string. "
            "Provide the raw receipt content to parse."
        )

    if not today_date or not today_date.strip():
        raise ValueError(
            "today_date must be a non-empty ISO-8601 date string (YYYY-MM-DD). "
            "Pass datetime.date.today().isoformat() if no specific date is needed."
        )

    logger.info(
        "Starting receipt parse | fallback_date=%s | input_chars=%d",
        today_date,
        len(receipt_text),
    )

    # ── Build chain (lightweight, no network call yet) ────────────────────────
    try:
        chain = _build_chain()
    except Exception as exc:
        logger.error("Failed to build LangChain chain: %s", exc, exc_info=True)
        raise RuntimeError(
            f"Could not initialise the LLM chain. "
            f"Verify that GOOGLE_API_KEY is set and langchain_google_genai is installed. "
            f"Underlying error: {exc}"
        ) from exc

    # ── Invoke LLM ────────────────────────────────────────────────────────────
    try:
        payload: ReceiptPayload = await chain.ainvoke(
            {
                "today_date": today_date,
                "receipt_text": receipt_text,
            }
        )
    except ValueError as exc:
        # Raised by Pydantic when the LLM output violates schema constraints.
        logger.error(
            "Schema validation failed for parsed receipt | error=%s", exc, exc_info=True
        )
        raise RuntimeError(
            f"The LLM returned a response that failed Pydantic validation. "
            f"This usually means the receipt text was too ambiguous or malformed. "
            f"Validation error: {exc}"
        ) from exc
    except Exception as exc:
        # Covers: network timeouts, auth errors, quota exhaustion, etc.
        logger.error(
            "Unexpected error during receipt parsing | error=%s", exc, exc_info=True
        )
        raise

    # ── Post-parse sanity check ───────────────────────────────────────────────
    if not payload.items:
        logger.warning(
            "Parser returned zero items. Receipt may be empty, "
            "contain only non-product lines, or be in an unsupported format. "
            "receipt_preview=%r",
            receipt_text[:200],
        )

    logger.info(
        "Receipt parsed successfully | items_found=%d | date_of_purchase=%s",
        len(payload.items),
        payload.date_of_purchase,
    )

    return payload


def parse_receipt_text_local(receipt_text: str, today_date: str) -> ReceiptPayload:
    """
    A fast, synchronous local parser that processes receipt lines and lists
    using regex, bypassing LLM API calls completely.
    """
    import re
    from app.models import ReceiptItem, ReceiptPayload

    if not receipt_text or not receipt_text.strip():
        raise ValueError("receipt_text must be a non-empty string.")
        
    items_map: dict[str, ReceiptItem] = {}
    
    # Split text into lines, then split those lines by commas to handle comma-separated lists
    segments: list[str] = []
    for line in receipt_text.splitlines():
        for segment in line.split(","):
            segments.append(segment.strip())
            
    # Unit mapping for normalization
    unit_map = {
        "lbs": "lbs", "lb": "lbs", "pound": "lbs", "pounds": "lbs",
        "kg": "kg", "kilogram": "kg", "kilograms": "kg",
        "g": "g", "gram": "g", "grams": "g",
        "oz": "oz", "ounce": "oz", "ounces": "oz",
        "pcs": "pcs", "piece": "pcs", "pieces": "pcs", "pc": "pcs", "item": "pcs", "items": "pcs",
        "l": "L", "litre": "L", "litres": "L", "liter": "L", "liters": "L",
        "ml": "ml", "millilitre": "ml", "millilitres": "ml", "milliliter": "ml", "milliliters": "ml",
        "gallon": "L", "gallons": "L", "gal": "L",
        "carton": "pcs", "cartons": "pcs",
        "head": "pcs", "heads": "pcs",
        "cup": "pcs", "cups": "pcs",
        "tbsp": "pcs", "tsp": "pcs"
    }

    # Regex to match a quantity and a unit
    unit_pattern = re.compile(
        r"(?:^|\s)(?P<qty>\d+(?:\.\d+)?)\s*(?P<unit>lbs|lb|pounds|pound|kg|kilograms|kilogram|g|grams|gram|oz|ounces|ounce|pcs|pieces|piece|pc|l|litre|litres|liter|liters|ml|millilitre|millilitres|milliliter|milliliters|gallon|gallons|gal|carton|cartons|head|heads|cup|cups|tbsp|tsp)(?:\s+|$)",
        re.IGNORECASE
    )
    
    # Regex to match just a number at the start or end of the string
    num_start_pattern = re.compile(r"^\s*(?P<qty>\d+(?:\.\d+)?)\s+", re.IGNORECASE)
    num_end_pattern = re.compile(r"\s+(?P<qty>\d+(?:\.\d+)?)\s*$", re.IGNORECASE)

    for seg in segments:
        if not seg:
            continue
            
        # Ignore lines with receipt headers/totals/noise
        seg_lower = seg.lower()
        if any(term in seg_lower for term in ["total", "tax", "subtotal", "payment", "cashier", "date", "store", "receipt"]):
            continue
            
        qty = 1.0
        unit = "pcs"
        item_name = seg
        
        # 1. Try matching quantity and unit
        match = unit_pattern.search(item_name)
        if match:
            qty = float(match.group("qty"))
            raw_unit = match.group("unit").lower()
            unit = unit_map.get(raw_unit, "pcs")
            # Remove the match from name
            start, end = match.span()
            item_name = item_name[:start] + " " + item_name[end:]
        else:
            # 2. Try matching a bare number at start
            match_start = num_start_pattern.search(item_name)
            if match_start:
                qty = float(match_start.group("qty"))
                unit = "pcs"
                item_name = item_name[match_start.end():]
            else:
                # 3. Try matching a bare number at end
                match_end = num_end_pattern.search(item_name)
                if match_end:
                    qty = float(match_end.group("qty"))
                    unit = "pcs"
                    item_name = item_name[:match_end.start()]
                    
        # Clean the item name: remove bullet markers, extra spaces
        item_name = re.sub(r"^[\s\-\*\•\+]+", "", item_name)
        item_name = re.sub(r"[\s\-\*\•\+]+$", "", item_name)
        item_name = re.sub(r"\s+", " ", item_name).strip().lower()
        
        if not item_name:
            continue
            
        # Deduplication
        if item_name in items_map:
            existing = items_map[item_name]
            items_map[item_name] = ReceiptItem(
                item_name=item_name,
                count=existing.count + qty,
                unit=existing.unit,
                purchase_date=today_date
            )
        else:
            items_map[item_name] = ReceiptItem(
                item_name=item_name,
                count=qty,
                unit=unit,
                purchase_date=today_date
            )
            
    return ReceiptPayload(
        items=list(items_map.values()),
        date_of_purchase=today_date
    )

