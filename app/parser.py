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

_MODEL_NAME = "gemini-1.5-pro"  # gemini-pro is deprecated as of langchain-google-genai v4.x
_TEMPERATURE = 0.0              # Zero temperature → deterministic, schema-compliant extraction
_MAX_RETRIES = 2                # Retry transient API failures before raising


def _build_llm() -> ChatGoogleGenerativeAI:
    """
    Instantiate the ChatGoogleGenerativeAI client.

    API key is read automatically from the GOOGLE_API_KEY environment variable
    by the langchain_google_genai package. Set it in your environment or a
    .env file loaded at application startup (e.g., via python-dotenv in main.py).
    """
    return ChatGoogleGenerativeAI(
        model=_MODEL_NAME,
        temperature=_TEMPERATURE,
        max_retries=_MAX_RETRIES,
        # convert_system_message_to_human=True is NOT set here because
        # Gemini Pro supports native system messages via langchain ≥ 0.2.
    )


# ─────────────────────────────────────────────────────────────────────────────
# Prompt Template
# ─────────────────────────────────────────────────────────────────────────────

# The system prompt is parameterized with {today_date} so the LLM always has
# a concrete fallback date without hard-coding any value at import time.
_SYSTEM_PROMPT_TEMPLATE = """\
You are an isolated data-extraction compiler. Your sole objective is to \
convert unstructured grocery receipt text or markdown syntax into a \
standardized JSON structure matching the required schema.

Operational Directives:

1. Normalize item descriptions: Clean raw or noisy storefront text into \
clean, descriptive, human-readable item names \
(e.g., change "BH Fresh Mozzarella Ball" into "mozzarella ball"). \
All item_name values must be lowercase.

2. Quantity Coercion: Extract fractional weights or integers into the \
appropriate `count` fields. Ensure values match common units: \
'lbs', 'kg', 'g', 'oz', 'pcs', 'L', 'ml'. \
When the unit is ambiguous, infer from context \
(e.g., produce sold by weight → 'lbs'; packaged goods → 'pcs').

3. Temporal Mapping: Isolate the purchase date if printed on the receipt. \
Format it as ISO-8601 (YYYY-MM-DD). \
If no explicit date is present, default to the context-provided baseline: {today_date}. \
Apply this same date to every ReceiptItem.purchase_date and to date_of_purchase.

4. Deduplication: If the same item appears on multiple lines \
(e.g., bought twice), merge them into a single ReceiptItem by summing `count`.

5. Exclusions: Ignore non-product lines such as subtotals, taxes, \
loyalty points, cashier names, store addresses, and payment method rows.

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
        SystemMessage(content=_SYSTEM_PROMPT_TEMPLATE),
        HumanMessage(content=_HUMAN_PROMPT_TEMPLATE),
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
    llm = _build_llm()
    structured_llm = llm.with_structured_output(ReceiptPayload)

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
