from fastapi import APIRouter
from db import get_connection
from model import embed

router = APIRouter()

@router.post("/insert")
def insert_product(id: int, title: str, category: str):
    vector = embed(f"{title} {category}")
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO products (id, title, category, embedding) VALUES (%s, %s, %s, %s) "
        "ON CONFLICT (id) DO UPDATE SET title = EXCLUDED.title, category = EXCLUDED.category, embedding = EXCLUDED.embedding",
        (id, title, category, vector)
    )
    conn.commit()
    cur.close()
    conn.close()
    return {"status": "ok"}