from fastapi import APIRouter
from db import get_connection
from model import embed

router = APIRouter()

@router.post("/insert")
def insert_product(id: int, title: str, category: str):
    # Generate the embedding vector
    vector = embed(f"{title} {category}")
    
    # Generate the tsvector for full-text search
    tsv = f"to_tsvector('simple', %s || ' ' || %s)"
    
    # Connect to the database
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO products (id, title, category, embedding, tsv)
        VALUES (%s, %s, %s, %s, to_tsvector('simple', %s || ' ' || %s))
        ON CONFLICT (id)
        DO UPDATE SET 
            title = EXCLUDED.title, 
            category = EXCLUDED.category, 
            embedding = EXCLUDED.embedding,
            tsv = EXCLUDED.tsv
        """,
        (id, title, category, vector, title, category)
    )
    conn.commit()
    cur.close()
    conn.close()
    return {"status": "ok"}