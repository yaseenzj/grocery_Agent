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
from app.database import init_db, close_db_pool
from app.expiration import enrich_inventory_lifespans
from app.inventory_manager import get_current_inventory, process_consumption_event, update_inventory_stock, remove_inventory_item
from app.models import InventoryPayload, Recipe
from app.parser import parse_receipt_text, parse_receipt_text_local
from app.recipe_advisor import recommend_optimized_meals, precompute_and_cache_recipes, read_cached_recipes

def _configure_logging() -> None:
    level_name = os.getenv('LOG_LEVEL', 'INFO').upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format='%(asctime)s  %(levelname)-8s  %(name)s — %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

@asynccontextmanager
async def _lifespan(app: FastAPI):
    load_dotenv()
    _configure_logging()
    logger = logging.getLogger('main')
    api_key_set = bool(os.getenv('GOOGLE_API_KEY'))
    logger.info('Grocery Lifecycle Tracker starting up | GOOGLE_API_KEY_set=%s', api_key_set)
    if not api_key_set:
        logger.warning('GOOGLE_API_KEY is not set. All LLM-backed routes will fail. Create a .env file with GOOGLE_API_KEY=<your-key> or export it in your shell.')
    try:
        await init_db()
    except Exception as exc:
        logger.critical('Failed to initialize database on startup | error=%s', exc, exc_info=True)
    yield
    await close_db_pool()
    logger.info('Grocery Lifecycle Tracker shutting down.')
app = FastAPI(title='Grocery Lifecycle Tracker', summary='An agentic grocery assistant powered by Gemini Pro and LangChain.', description='A four-stage AI pipeline that ingests grocery receipts, enriches them with expiration data, tracks consumption events in a stateful inventory, and generates prioritized recipe recommendations to minimize food waste.', version='1.0.0', lifespan=_lifespan)
app.add_middleware(CORSMiddleware, allow_origins=['http://localhost:5173', 'http://127.0.0.1:5173'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])
logger = logging.getLogger(__name__)

class ReceiptUploadRequest(BaseModel):
    receipt_text: str = Field(..., min_length=1, description='Raw, unstructured grocery receipt text. Acceptable formats include plain text, OCR output, markdown tables, or bullet lists. Must not be empty.', examples=['Whole Milk 1 gal\nChicken Breast 2 lbs\nBroccoli 1 head\nTotal: $14.72'])
    today_date: str = Field(default_factory=lambda: date.today().isoformat(), description="Fallback purchase date in ISO-8601 format (YYYY-MM-DD). Applied to any receipt item that lacks an explicit date. Defaults to the server's current date if omitted.", examples=['2025-07-01'])

class ConsumeRequest(BaseModel):
    statement: str = Field(..., min_length=1, description="Free-form user statement describing grocery consumption. Examples: 'I used 1 lb of onions', 'We finished half the carton of eggs', 'Used about 200g of spinach for the salad'.", examples=['I used 1 lb of onions'])

class RecipeListResponse(BaseModel):
    recipes: list[Recipe] = Field(..., description='Prioritized list of Recipe recommendations based on the current inventory. Sorted by inventory_match_score descending — the most achievable recipe first. Empty list if the inventory is fully consumed or empty.')
    count: int = Field(..., description='Total number of recipes returned in this response.')

class ErrorDetail(BaseModel):
    error: str = Field(..., description='Short machine-readable error category.')
    message: str = Field(..., description='Human-readable description of what went wrong.')
    detail: Any = Field(default=None, description='Optional structured detail (validation errors, etc.).')

@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    logger.warning('Request validation failed | path=%s | errors=%s', request.url.path, exc.errors())
    return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content=ErrorDetail(error='request_validation_error', message='The request body contains invalid or missing fields. Inspect `detail` for per-field error information.', detail=exc.errors()).model_dump())

@app.exception_handler(ValidationError)
async def pydantic_validation_exception_handler(request: Request, exc: ValidationError) -> JSONResponse:
    logger.error('Internal schema validation failed | path=%s | errors=%s', request.url.path, exc.errors(), exc_info=True)
    return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=ErrorDetail(error='schema_validation_error', message='An internal data validation error occurred. The AI model may have returned a response that violates the expected schema. Inspect `detail` for the specific field constraints that failed.', detail=exc.errors()).model_dump())

@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError) -> JSONResponse:
    logger.error('Pipeline RuntimeError | path=%s | error=%s', request.url.path, exc, exc_info=True)
    status_code = status.HTTP_502_BAD_GATEWAY
    error_type = 'pipeline_error'
    if 'Database' in str(exc):
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        error_type = 'database_error'
    return JSONResponse(status_code=status_code, content=ErrorDetail(error=error_type, message=str(exc), detail=None).model_dump())

@app.post('/receipt/upload', response_model=InventoryPayload, status_code=status.HTTP_200_OK, summary='Upload a grocery receipt', description='Accepts raw receipt text and runs the full ingestion pipeline: **parse** → **enrich with expiration dates** → **merge into inventory**. Returns the updated inventory snapshot after the new items are persisted.', tags=['Receipt'])
async def upload_receipt(body: ReceiptUploadRequest, background_tasks: BackgroundTasks) -> InventoryPayload:
    logger.info('POST /receipt/upload | date=%s | input_chars=%d', body.today_date, len(body.receipt_text))
    receipt_payload = parse_receipt_text_local(receipt_text=body.receipt_text, today_date=body.today_date)
    logger.info('Stage 1 complete (local parsing) | items_parsed=%d', len(receipt_payload.items))
    inventory_payload = await enrich_inventory_lifespans(receipt=receipt_payload)
    logger.info('Stage 2 complete | items_enriched=%d', len(inventory_payload.items))
    updated_inventory = await update_inventory_stock(incoming_stock=inventory_payload)
    logger.info('POST /receipt/upload complete | total_inventory_items=%d', len(updated_inventory.items))
    background_tasks.add_task(precompute_and_cache_recipes)
    logger.info('Stage 4 complete | scheduled background recipe precomputation')
    return updated_inventory

@app.post('/inventory/consume', response_model=InventoryPayload, status_code=status.HTTP_200_OK, summary='Record a consumption event', description="Accepts a natural-language consumption statement from the user (e.g., *'I used 1 lb of onions'*). Gemini Pro extracts the structured intent, debits the inventory using FIFO batch ordering, clamps any over-consumption to zero, and persists the result. Returns the updated inventory snapshot.", tags=['Inventory'])
async def consume_inventory(body: ConsumeRequest, background_tasks: BackgroundTasks) -> InventoryPayload:
    logger.info('POST /inventory/consume | statement=%r', body.statement)
    updated_inventory = await process_consumption_event(raw_user_speech=body.statement)
    logger.info('POST /inventory/consume complete | total_inventory_items=%d', len(updated_inventory.items))
    background_tasks.add_task(precompute_and_cache_recipes)
    logger.info('POST /inventory/consume | scheduled background recipe precomputation')
    return updated_inventory

@app.get('/recipes/recommend', response_model=RecipeListResponse, status_code=status.HTTP_200_OK, summary='Get prioritized recipe recommendations', description='Reads the current inventory, excludes exhausted items, sorts remaining ingredients by expiration urgency, and calls Gemini Pro to generate structured recipe recommendations. Recipes are ranked by `inventory_match_score` (highest first). Returns an empty `recipes` list if the inventory is fully consumed.', tags=['Recipes'])
async def get_recipe_recommendations() -> RecipeListResponse:
    logger.info('GET /recipes/recommend | fetching cached recipe recommendations.')
    recipes = await read_cached_recipes()
    if not recipes:
        logger.info('GET /recipes/recommend | Cache empty/missing, performing live fallback generation.')
        recipes = await recommend_optimized_meals()
    logger.info('GET /recipes/recommend complete | recipe_count=%d', len(recipes))
    return RecipeListResponse(recipes=recipes, count=len(recipes))

@app.delete('/inventory/item/{item_name}', response_model=InventoryPayload, status_code=status.HTTP_200_OK, summary='Remove an item from inventory', description='Removes all batches of the specified item name from the inventory.', tags=['Inventory'])
async def remove_item(item_name: str, background_tasks: BackgroundTasks) -> InventoryPayload:
    logger.info('DELETE /inventory/item/%s | removing item.', item_name)
    updated_inventory = await remove_inventory_item(item_name)
    background_tasks.add_task(precompute_and_cache_recipes)
    return updated_inventory

@app.get('/inventory', response_model=InventoryPayload, status_code=status.HTTP_200_OK, summary='Get current inventory snapshot', description='Returns the current state of the fridge inventory.', tags=['Inventory'])
async def get_inventory() -> InventoryPayload:
    logger.info('GET /inventory | fetching current inventory.')
    return await get_current_inventory()

@app.get('/health', status_code=status.HTTP_200_OK, summary='Health check', description='Returns a simple liveness signal. Does not exercise any pipeline modules.', tags=['System'], include_in_schema=True)
async def health_check() -> dict[str, str]:
    return {'status': 'ok', 'service': 'grocery-lifecycle-tracker'}