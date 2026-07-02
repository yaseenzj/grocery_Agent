# Grocery Assistant

An intelligent, agentic grocery management system powered by Gemini Pro and LangChain. This application automates the journey from a grocery receipt to a home-cooked meal by extracting items, tracking expiration dates, managing live inventory in PostgreSQL, and recommending optimized recipes to minimize food waste.

## Features

1. **Receipt Parsing**: Extracts structured item data (names, quantities, units) from raw receipt text or markdown using local regex optimizations and Gemini Pro.
2. **Expiration Estimation**: Calculates expiration dates using a local knowledge base with an LLM fallback for unknown items.
3. **Inventory Tracking (PostgreSQL)**: Understands natural language consumption statements (e.g., *"I used 1 lb of onions"*) and automatically debits the correct items using FIFO ordering with robust database transaction safety.
4. **Recipe Recommendation**: Generates realistic recipes based on your available inventory, prioritizing ingredients that are closest to their expiration date, and caches them in the database for instant retrieval.
5. **Interactive CLI**: Comes with a built-in terminal interface (`cli.py`) for managing your fridge directly from the command line.

## Architecture

The system is built as a modular FastAPI application backed by a PostgreSQL database:

| Module | Responsibility |
|---|---|
| `app/parser.py` | LangChain structured output for receipt ingestion |
| `app/expiration.py` | Hybrid local DB + LLM fallback for shelf-life estimation |
| `app/inventory_manager.py` | Asynchronous PostgreSQL persistence, FIFO consumption debiting |
| `app/recipe_advisor.py` | Expiry-prioritized recipe generation engine and caching |
| `app/models.py` | Pydantic v2 schemas for all data contracts |
| `app/database.py` | asyncpg connection pooling and automated table migrations |
| `cli.py` | Interactive terminal interface |

State is persisted reliably to PostgreSQL, guaranteeing data integrity and fast concurrent access.

## Prerequisites

- Python 3.11 or higher
- PostgreSQL database running locally or remotely
- A Google Gemini API Key — get a free one at [Google AI Studio](https://aistudio.google.com/app/apikey)

## Installation & Setup

**1. Clone the Repository**
```bash
git clone https://github.com/yaseenzj/grocery_Agent.git
cd grocery_Agent
```

**2. Create a Virtual Environment**
```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# macOS / Linux:
source venv/bin/activate
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**4. Configure Environment Variables**

Create a `.env` file in the root directory (this file is gitignored and will never be committed):
```env
GOOGLE_API_KEY=your_api_key_here
DATABASE_URL=postgresql://postgres:password@localhost:5432/grocery_db
```

## Running the Application

You can use the system via the interactive CLI or the REST API.

**Interactive CLI:**
```bash
python cli.py
```
This will automatically initialize the database schema and provide an interactive terminal menu to manage your groceries and view recipes.

**REST API Server:**
```bash
uvicorn main:app --reload --port 8000
```
The server starts at `http://localhost:8000`. On startup, it automatically configures the PostgreSQL connection pool and runs any necessary migrations.

## API Endpoints

Interactive documentation is available at **[http://localhost:8000/docs](http://localhost:8000/docs)**.

### `POST /receipt/upload`
Upload a raw receipt string to parse items, enrich with expiration dates, and persist to inventory.

**Request body:**
```json
{
  "receipt_text": "Bought 2 lbs of chicken breast, 1 gallon of whole milk, 1 head of broccoli.",
  "today_date": "2025-07-01"
}
```

### `POST /inventory/consume`
Tell the system what you consumed in plain English.

**Request body:**
```json
{
  "statement": "I used half a gallon of milk for baking."
}
```

### `GET /recipes/recommend`
Get recipe recommendations based on current inventory, prioritizing ingredients that expire soonest. Automatically falls back to the database cache for instant responses.

### `GET /health`
Simple liveness check — returns `{ "status": "ok", "service": "grocery-lifecycle-tracker" }`.

## Security Notes

The following are excluded from version control via `.gitignore`:
- `.env` — contains your private API key and database credentials
- `__pycache__/` — Python bytecode cache
- `venv/` — local dependency installations