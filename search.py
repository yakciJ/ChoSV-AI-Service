import json
from fastapi import APIRouter, Query
from db import get_connection
from model import embed
import numpy as np
from underthesea import word_tokenize, pos_tag

router = APIRouter()

def extract_keywords(query):
    tokens = word_tokenize(query, format="text").split()
    nouns = [w for w, p in pos_tag(query) if p == "N"]
    return {
        "full": query,
        "tokens": tokens,
        "nouns": nouns
    }

@router.get("/search")
def search_products(q: str = Query(...)):
    kws = extract_keywords(q)
    q_vec = embed(q)

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, title, category, embedding FROM products")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    
    results = []
    for r in rows:
        pid, title, cate, emb = r
        try:
            # Parse the string representation of the embedding into a Python list
            emb = np.array(json.loads(emb), dtype=float)  # Use `ast.literal_eval(emb)` as an alternative
            score = float(np.dot(q_vec, emb) / (np.linalg.norm(q_vec) * np.linalg.norm(emb)))
            # Bonus if keyword matches
            bonus = sum(1 for kw in kws["nouns"] if kw in title.lower() or kw in cate.lower())
            total = 0.8 * score + 0.2 * bonus
            results.append({
                "id": pid, "title": title, "category": cate, "score": total
            })
        except Exception as e:
            print(f"Error processing row {r}: {e}")
            continue  # Skip problematic rows

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:20]