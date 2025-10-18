from fastapi import APIRouter
from db import get_async_connection
from model import embed
import asyncio

router = APIRouter()

@router.post("/insert")
async def insert_product(id: int, title: str, category: str):
    # Generate the embedding vector
    vector = await asyncio.to_thread(embed, f"{title} {category}")
    vector_str = "[" + ", ".join(map(str, vector)) + "]"
    # Connect to the database
    conn = await get_async_connection()
    try:
        await conn.execute(
            """
            INSERT INTO products (id, title, category, embedding, tsv)
            VALUES ($1, $2, $3, $4::vector, to_tsvector('simple', $5 || ' ' || $6))
            ON CONFLICT (id)
            DO UPDATE SET 
                title = EXCLUDED.title, 
                category = EXCLUDED.category, 
                embedding = EXCLUDED.embedding,
                tsv = EXCLUDED.tsv
            """,
            id, title, category, vector_str, title, category
        )
    finally:
        await conn.close()
    return {"status": "ok"}

@router.put("/update")
async def update_product(id: int, title: str, category: str):
    # Generate the embedding vector
    vector = await asyncio.to_thread(embed, f"{title} {category}")
    vector_str = "[" + ", ".join(map(str, vector)) + "]"

    # Connect to the database
    conn = await get_async_connection()
    try:
        await conn.execute(
            """
            UPDATE products
            SET title = $2, category = $3, embedding = $4, tsv = to_tsvector('simple', $2 || ' ' || $3)
            WHERE id = $1
            """,
            id, title, category, vector_str
        )
    finally:
        await conn.close()

    return {"status": "ok"}

@router.delete("/delete")
async def delete_product(id: int):
    # Connect to the database
    conn = await get_async_connection()
    try:
        await conn.execute(
            """
            DELETE FROM products
            WHERE id = $1
            """,
            id
        )
    finally:
        await conn.close()

    return {"status": "ok"}