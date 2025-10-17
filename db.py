import asyncpg
import os

async def get_async_connection():
    conn = await asyncpg.connect(
        database=os.getenv("PG_DB", "ChoSV"),
        user=os.getenv("PG_USER", "postgres"),
        password=os.getenv("PG_PASS", "1234"),
        host=os.getenv("PG_HOST", "localhost"),
        port=os.getenv("PG_PORT", 5432),
    )
    return conn