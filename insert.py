from fastapi import APIRouter
from db import get_async_connection
from model import embed
import asyncio

router = APIRouter()

@router.post("/insert")
async def insert_product(id: int, title: str, category: str):
    # Generate the embedding vector
    vector = await asyncio.to_thread(embed, f"{title} {category}")
    
    # Generate the tsvector for full-text search
    # tsv = f"to_tsvector('simple', %s || ' ' || %s)"
    
    # Connect to the database
    conn = await get_async_connection()
    try:
        await conn.execute(
            """
            INSERT INTO products (id, title, category, embedding, tsv)
            VALUES ($1, $2, $3, $4, to_tsvector('simple', $5 || ' ' || $6))
            ON CONFLICT (id)
            DO UPDATE SET 
                title = EXCLUDED.title, 
                category = EXCLUDED.category, 
                embedding = EXCLUDED.embedding,
                tsv = EXCLUDED.tsv
            """,
            id, title, category, vector, title, category
        )
    finally:
        await conn.close()
    return {"status": "ok"}