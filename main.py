"""
main.py
─────────────────────────────────────────────────────────────────────────────
Interface Orchestration Layer — Grocery Lifecycle Tracker API.

Maps the four internal pipeline modules to three public FastAPI routes:

  POST /receipt/upload      → parse_receipt_text → enrich_inventory_lifespans
                              → update_inventory_stock → InventoryPayload
  POST /inventory/consume   → process_consumption_event → InventoryPayload
  GET  /recipes/recommend   → recommend_optimized_meals → list[Recipe]

Design notes:
  - Every route is a thin delegation wrapper. Business logic lives exclusively
    in the pipeline modules (parser, expiration, inventory_manager, recipe_advisor).
  - Custom exception handlers intercept Pydantic v2 ValidationError and
    FastAPI RequestValidationError and return clean, structured JSON error bodies
    instead of framework defaults.
  - GOOGLE_API_KEY is loaded from a .env file at startup via python-dotenv.
    The key never appears in route logic — only the LLM clients read it via the
    GOOGLE_API_KEY environment variable implicitly.
  - Logging is configured once at startup (INFO level to stdout by default).

Run locally:
    uvicorn main:app --reload --port 8000
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from datetime import date
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, Request, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from app.expiration import enrich_inventory_lifespans
from app.inventory_manager import process_consumption_event, update_inventory_stock, remove_inventory_item
from app.models import InventoryPayload, Recipe
from app.parser import parse_receipt_text, parse_receipt_text_local
from app.recipe_advisor import recommend_optimized_meals, precompute_and_cache_recipes, read_cached_recipes



# ─────────────────────────────────────────────────────────────────────────────
# Startup: environment & logging
# ─────────────────────────────────────────────────────────────────────────────

def _configure_logging() -> None:
    """
    Set up a simple, readable logging format for the entire application.

    Uses INFO as the root level so pipeline debug logs stay quiet by default.
    Set the LOG_LEVEL environment variable (e.g., LOG_LEVEL=DEBUG) to increase
    verbosity without code changes.
    """
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """
    FastAPI lifespan handler — runs once at startup before any request is served.

    Loads environment variables from a local .env file so that GOOGLE_API_KEY
    is available to all LangChain / Gemini clients without requiring the caller
    to export it manually in the shell.
    """
    # Load .env before any module reads environment variables
    load_dotenv()
    _configure_logging()

    logger = logging.getLogger("main")
    api_key_set = bool(os.getenv("GOOGLE_API_KEY"))
    logger.info(
        "Grocery Lifecycle Tracker starting up | GOOGLE_API_KEY_set=%s",
        api_key_set,
    )
    if not api_key_set:
        logger.warning(
            "GOOGLE_API_KEY is not set. All LLM-backed routes will fail. "
            "Create a .env file with GOOGLE_API_KEY=<your-key> or export it in your shell."
        )

    yield  # Application is running

    logger.info("Grocery Lifecycle Tracker shutting down.")


# ─────────────────────────────────────────────────────────────────────────────
# Application Instance
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Grocery Lifecycle Tracker",
    summary="An agentic grocery assistant powered by Gemini Pro and LangChain.",
    description=(
        "A four-stage AI pipeline that ingests grocery receipts, enriches them "
        "with expiration data, tracks consumption events in a stateful inventory, "
        "and generates prioritized recipe recommendations to minimize food waste."
    ),
    version="1.0.0",
    lifespan=_lifespan,
)

# Add CORS middleware so Framer (or any frontend) can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response Schemas
# ─────────────────────────────────────────────────────────────────────────────

class ReceiptUploadRequest(BaseModel):
    """
    Request body for POST /receipt/upload.

    Accepts raw receipt text (OCR output, typed notes, markdown) and an optional
    fallback date used when the receipt itself omits a purchase date.
    """

    receipt_text: str = Field(
        ...,
        min_length=1,
        description=(
            "Raw, unstructured grocery receipt text. Acceptable formats include "
            "plain text, OCR output, markdown tables, or bullet lists. "
            "Must not be empty."
        ),
        examples=[
            "Whole Milk 1 gal\nChicken Breast 2 lbs\nBroccoli 1 head\nTotal: $14.72"
        ],
    )

    today_date: str = Field(
        default_factory=lambda: date.today().isoformat(),
        description=(
            "Fallback purchase date in ISO-8601 format (YYYY-MM-DD). "
            "Applied to any receipt item that lacks an explicit date. "
            "Defaults to the server's current date if omitted."
        ),
        examples=["2025-07-01"],
    )


class ConsumeRequest(BaseModel):
    """
    Request body for POST /inventory/consume.

    Accepts a free-form natural-language statement describing what the user
    consumed. The Gemini extraction layer converts this to a structured debit.
    """

    statement: str = Field(
        ...,
        min_length=1,
        description=(
            "Free-form user statement describing grocery consumption. "
            "Examples: 'I used 1 lb of onions', 'We finished half the carton of eggs', "
            "'Used about 200g of spinach for the salad'."
        ),
        examples=["I used 1 lb of onions"],
    )


class RecipeListResponse(BaseModel):
    """
    Response envelope for GET /recipes/recommend.

    Wraps the raw list[Recipe] in a named field so the top-level JSON response
    is always an object (never a bare array), which is safer for API consumers
    and more forward-compatible for versioning.
    """

    recipes: list[Recipe] = Field(
        ...,
        description=(
            "Prioritized list of Recipe recommendations based on the current inventory. "
            "Sorted by inventory_match_score descending — the most achievable recipe first. "
            "Empty list if the inventory is fully consumed or empty."
        ),
    )

    count: int = Field(
        ...,
        description="Total number of recipes returned in this response.",
    )


class ErrorDetail(BaseModel):
    """Standardized error body returned by all custom exception handlers."""

    error: str = Field(..., description="Short machine-readable error category.")
    message: str = Field(..., description="Human-readable description of what went wrong.")
    detail: Any = Field(default=None, description="Optional structured detail (validation errors, etc.).")


# ─────────────────────────────────────────────────────────────────────────────
# Custom Exception Handlers
# ─────────────────────────────────────────────────────────────────────────────

@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """
    Handle malformed or missing request fields (FastAPI input validation layer).

    Triggered when a request body fails Pydantic validation before it reaches
    the route function — e.g., missing required fields, wrong types, constraint
    violations on ReceiptUploadRequest or ConsumeRequest.

    Returns HTTP 422 with a clean, structured error body instead of the default
    FastAPI verbose format.
    """
    logger.warning(
        "Request validation failed | path=%s | errors=%s",
        request.url.path,
        exc.errors(),
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorDetail(
            error="request_validation_error",
            message=(
                "The request body contains invalid or missing fields. "
                "Inspect `detail` for per-field error information."
            ),
            detail=exc.errors(),
        ).model_dump(),
    )


@app.exception_handler(ValidationError)
async def pydantic_validation_exception_handler(
    request: Request,
    exc: ValidationError,
) -> JSONResponse:
    """
    Handle Pydantic v2 ValidationError raised inside route logic.

    Triggered when the LLM returns data that fails schema validation after
    reaching Python — e.g., a Recipe with a negative inventory_match_score,
    or an InventoryItem with a malformed expiration_date.

    Returns HTTP 500 (the failure is internal — the user's input was valid)
    with a clean error body surfacing the Pydantic error list.
    """
    logger.error(
        "Internal schema validation failed | path=%s | errors=%s",
        request.url.path,
        exc.errors(),
        exc_info=True,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorDetail(
            error="schema_validation_error",
            message=(
                "An internal data validation error occurred. The AI model may have "
                "returned a response that violates the expected schema. "
                "Inspect `detail` for the specific field constraints that failed."
            ),
            detail=exc.errors(),
        ).model_dump(),
    )


@app.exception_handler(RuntimeError)
async def runtime_error_handler(
    request: Request,
    exc: RuntimeError,
) -> JSONResponse:
    """
    Handle RuntimeError raised by pipeline modules.

    All pipeline modules (parser, expiration, inventory_manager, recipe_advisor)
    wrap their internal failures in RuntimeError with a descriptive message.
    This handler surfaces that message as a clean HTTP 502 (bad gateway)
    — indicating that an upstream service (Gemini API) or I/O operation failed.
    """
    logger.error(
        "Pipeline RuntimeError | path=%s | error=%s",
        request.url.path,
        exc,
        exc_info=True,
    )
    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY,
        content=ErrorDetail(
            error="pipeline_error",
            message=str(exc),
            detail=None,
        ).model_dump(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Route 1 — POST /receipt/upload
# ─────────────────────────────────────────────────────────────────────────────

@app.post(
    "/receipt/upload",
    response_model=InventoryPayload,
    status_code=status.HTTP_200_OK,
    summary="Upload a grocery receipt",
    description=(
        "Accepts raw receipt text and runs the full ingestion pipeline: "
        "**parse** → **enrich with expiration dates** → **merge into inventory**. "
        "Returns the updated inventory snapshot after the new items are persisted."
    ),
    tags=["Receipt"],
)
async def upload_receipt(
    body: ReceiptUploadRequest,
    background_tasks: BackgroundTasks,
) -> InventoryPayload:
    """
    Full receipt ingestion pipeline in a single endpoint.

    Pipeline steps executed synchronously (each awaited in order):
      1. ``parse_receipt_text_local``  — Fast local regex parsing -> ReceiptPayload
      2. ``enrich_inventory_lifespans``— ReceiptPayload → InventoryPayload (+ expiration dates)
      3. ``update_inventory_stock``    — Merges into data/inventory.json, returns merged state
      4. Silently schedules recipe precomputation in background tasks.

    Args:
        body: Validated :class:`ReceiptUploadRequest` with ``receipt_text``
              and optional ``today_date`` fallback.
        background_tasks: FastAPI BackgroundTasks system.

    Returns:
        The merged :class:`~app.models.InventoryPayload` as now persisted on disk.
    """
    logger.info(
        "POST /receipt/upload | date=%s | input_chars=%d",
        body.today_date,
        len(body.receipt_text),
    )

    # Stage 1 — Parse raw receipt text locally into structured ReceiptPayload
    receipt_payload = parse_receipt_text_local(
        receipt_text=body.receipt_text,
        today_date=body.today_date,
    )
    logger.info("Stage 1 complete (local parsing) | items_parsed=%d", len(receipt_payload.items))

    # Stage 2 — Enrich each item with a computed expiration_date
    inventory_payload = await enrich_inventory_lifespans(receipt=receipt_payload)
    logger.info("Stage 2 complete | items_enriched=%d", len(inventory_payload.items))

    # Stage 3 — Merge into the persistent inventory store
    updated_inventory = await update_inventory_stock(incoming_stock=inventory_payload)
    logger.info(
        "POST /receipt/upload complete | total_inventory_items=%d",
        len(updated_inventory.items),
    )

    # Stage 4 — Trigger recipe precomputation in the background
    background_tasks.add_task(precompute_and_cache_recipes)
    logger.info("Stage 4 complete | scheduled background recipe precomputation")

    return updated_inventory



# ─────────────────────────────────────────────────────────────────────────────
# Route 2 — POST /inventory/consume
# ─────────────────────────────────────────────────────────────────────────────

@app.post(
    "/inventory/consume",
    response_model=InventoryPayload,
    status_code=status.HTTP_200_OK,
    summary="Record a consumption event",
    description=(
        "Accepts a natural-language consumption statement from the user "
        "(e.g., *'I used 1 lb of onions'*). "
        "Gemini Pro extracts the structured intent, debits the inventory using "
        "FIFO batch ordering, clamps any over-consumption to zero, and persists "
        "the result. Returns the updated inventory snapshot."
    ),
    tags=["Inventory"],
)
async def consume_inventory(
    body: ConsumeRequest,
    background_tasks: BackgroundTasks,
) -> InventoryPayload:
    """
    Natural-language consumption event → structured inventory debit.

    Delegates entirely to ``process_consumption_event``, which owns the full
    NL-extraction → debit → persist pipeline internally.

    Args:
        body: Validated :class:`ConsumeRequest` with the ``statement`` field.
        background_tasks: FastAPI BackgroundTasks system.

    Returns:
        The updated :class:`~app.models.InventoryPayload` as now persisted on disk.
        If the item mentioned is not found in inventory, the unchanged payload
        is returned (the function logs a warning but does not raise).
    """
    logger.info("POST /inventory/consume | statement=%r", body.statement)

    updated_inventory = await process_consumption_event(
        raw_user_speech=body.statement,
    )

    logger.info(
        "POST /inventory/consume complete | total_inventory_items=%d",
        len(updated_inventory.items),
    )

    # Trigger recipe precomputation in the background
    background_tasks.add_task(precompute_and_cache_recipes)
    logger.info("POST /inventory/consume | scheduled background recipe precomputation")

    return updated_inventory



# ─────────────────────────────────────────────────────────────────────────────
# Route 3 — GET /recipes/recommend
# ─────────────────────────────────────────────────────────────────────────────

@app.get(
    "/recipes/recommend",
    response_model=RecipeListResponse,
    status_code=status.HTTP_200_OK,
    summary="Get prioritized recipe recommendations",
    description=(
        "Reads the current inventory, excludes exhausted items, sorts remaining "
        "ingredients by expiration urgency, and calls Gemini Pro to generate "
        "structured recipe recommendations. "
        "Recipes are ranked by `inventory_match_score` (highest first). "
        "Returns an empty `recipes` list if the inventory is fully consumed."
    ),
    tags=["Recipes"],
)
async def get_recipe_recommendations() -> RecipeListResponse:
    """
    Inventory-aware, expiry-prioritized recipe recommendation.

    Delegates entirely to ``recommend_optimized_meals``, which owns the full
    filter → sort → LLM → score → rank pipeline internally.

    Returns:
        A :class:`RecipeListResponse` containing the ranked recipe list and
        the total recipe count. ``recipes`` is an empty list (HTTP 200, not 404)
        when the inventory is fully consumed — callers should use ``count == 0``
        to detect this state and prompt the user to restock.

    Raises:
        HTTP 502: If the Gemini API call fails after exhausting retries.
        HTTP 500: If the LLM response fails Recipe schema validation.
    """
    logger.info("GET /recipes/recommend | fetching cached recipe recommendations.")

    recipes = await read_cached_recipes()
    if not recipes:
        logger.info("GET /recipes/recommend | Cache empty/missing, performing live fallback generation.")
        recipes = await recommend_optimized_meals()

    logger.info(
        "GET /recipes/recommend complete | recipe_count=%d",
        len(recipes),
    )

    return RecipeListResponse(recipes=recipes, count=len(recipes))


# ─────────────────────────────────────────────────────────────────────────────
# Route 4 — DELETE /inventory/item/{item_name}
# ─────────────────────────────────────────────────────────────────────────────

@app.delete(
    "/inventory/item/{item_name}",
    response_model=InventoryPayload,
    status_code=status.HTTP_200_OK,
    summary="Remove an item from inventory",
    description="Removes all batches of the specified item name from the inventory.",
    tags=["Inventory"],
)
async def remove_item(
    item_name: str,
    background_tasks: BackgroundTasks,
) -> InventoryPayload:
    """
    Remove all batches of an item from the fridge by its exact name.
    
    Triggers recipe precomputation in the background after successful deletion.
    """
    logger.info("DELETE /inventory/item/%s | removing item.", item_name)
    updated_inventory = await remove_inventory_item(item_name)
    background_tasks.add_task(precompute_and_cache_recipes)
    return updated_inventory


# ─────────────────────────────────────────────────────────────────────────────
# Health Check
# ─────────────────────────────────────────────────────────────────────────────


@app.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="Health check",
    description="Returns a simple liveness signal. Does not exercise any pipeline modules.",
    tags=["System"],
    include_in_schema=True,
)
async def health_check() -> dict[str, str]:
    """Lightweight liveness probe for load balancers and deployment checks."""
    return {"status": "ok", "service": "grocery-lifecycle-tracker"}
