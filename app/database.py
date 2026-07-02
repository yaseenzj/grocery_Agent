import os
import logging
import asyncpg
from typing import Optional
logger = logging.getLogger(__name__)
_POOL: Optional[asyncpg.Pool] = None

async def get_db_pool() -> asyncpg.Pool:
    global _POOL
    if _POOL is None:
        db_url = os.getenv('DATABASE_URL')
        if not db_url:
            raise RuntimeError('DATABASE_URL environment variable is not set in .env. Please add DATABASE_URL=postgresql://user:password@host:port/dbname')
        try:
            logger.info('Initializing PostgreSQL database pool...')
            _POOL = await asyncpg.create_pool(dsn=db_url, min_size=1, max_size=10, timeout=10.0)
        except Exception as exc:
            logger.error('Failed to connect to PostgreSQL | error=%s', exc, exc_info=True)
            raise RuntimeError(f'Database connection failed: {exc}') from exc
    return _POOL

async def close_db_pool() -> None:
    global _POOL
    if _POOL is not None:
        logger.info('Closing PostgreSQL database pool...')
        await _POOL.close()
        _POOL = None

async def init_db() -> None:
    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            logger.info('Running database migrations/initialization...')
            await conn.execute('\n                CREATE TABLE IF NOT EXISTS inventory_items (\n                    id SERIAL PRIMARY KEY,\n                    item_name VARCHAR(255) NOT NULL,\n                    count DOUBLE PRECISION NOT NULL,\n                    unit VARCHAR(50) NOT NULL,\n                    purchase_date DATE NOT NULL,\n                    expiration_date DATE NOT NULL,\n                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP\n                );\n            ')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_inventory_items_name ON inventory_items(item_name);')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_inventory_items_expiration ON inventory_items(expiration_date);')
            await conn.execute('\n                CREATE TABLE IF NOT EXISTS recipes_cache (\n                    id SERIAL PRIMARY KEY,\n                    recipes_json TEXT NOT NULL,\n                    cached_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP\n                );\n            ')
            logger.info('Database initialization complete.')
    except Exception as exc:
        logger.error('Failed to initialize database | error=%s', exc, exc_info=True)
        raise RuntimeError(f'Database initialization failed: {exc}') from exc