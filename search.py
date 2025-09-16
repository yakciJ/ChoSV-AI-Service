"""
Hybrid search router for products (title + category) using:
- Vietnamese query processing (underthesea): full sentence, phrases, tokens
- PostgreSQL FTS (GIN) on products.tsv
- pgvector cosine similarity on products.embedding
- Match bonus: full +10, phrase +3, token +1
- Final score (normalized to [0,1]):
    total_score = ALPHA * vector_score + (1 - ALPHA) * normalized_match_bonus
  where:
    - vector_score ∈ [0,1] = 1 - (embedding <=> query_vec)
    - normalized_match_bonus ∈ [0,1] = match_bonus / max(match_bonus) over this query's candidates

This version:
- Uses your synchronous psycopg2 connection from db.get_connection()
- Uses your model.embed(text) from model.py (normalized embeddings)
- Exposes FastAPI router with GET /search endpoint returning JSON

Env vars:
- ALPHA            (default: 0.7)  weight for vector_score; (1-ALPHA) is weight for normalized match bonus
- CANDIDATE_LIMIT  (default: 1000) number of ranked rows to cache per query
- REDIS_URL        (default: redis://localhost:6379/0)
- CACHE_TTL        (default: 600)
- CACHE_MODEL_TAG  (default: "")
"""

from __future__ import annotations

import os
import re
import json
import hashlib
from typing import Any, Dict, List, Optional, Tuple

# FastAPI router
from fastapi import APIRouter, Query, HTTPException

# Vietnamese NLP
from underthesea import word_tokenize, pos_tag, chunk

# Your existing embedder
from model import embed as embed_query  # type: ignore

# Redis (sync) is optional
try:
    import redis  # type: ignore
except Exception:  # pragma: no cover
    redis = None  # type: ignore

# DB connection (psycopg2)
from db import get_connection  # type: ignore
from psycopg2.extras import RealDictCursor  # type: ignore


# =========================
# Config
# =========================

ALPHA = float(os.getenv("ALPHA", "0.7"))              # weight for vector_score
CANDIDATE_LIMIT = int(os.getenv("CANDIDATE_LIMIT", "1000"))
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))
CACHE_MODEL_TAG = os.getenv("CACHE_MODEL_TAG", "")

_CACHE_VERSION = "v2_norm"  # bump when changing cache schema



# =========================
# Helpers
# =========================

def normalize_query(q: str) -> str:
    return " ".join(q.strip().lower().split())


def split_query_vi(q: str, max_phrases: int = 10, max_tokens: int = 16) -> Tuple[str, List[str], List[str]]:
    """
    Returns (full_sentence, phrases, tokens)
    - tokens: unique, order-preserving, truncated to max_tokens
    - phrases: from chunk NP/VP/PP; fallback to bigrams if chunking fails
    """
    q_norm = normalize_query(q)

    # Tokens
    raw_tokens = word_tokenize(q_norm)
    seen = set()
    tokens: List[str] = []
    for t in raw_tokens:
        t = t.strip()
        if not t or t in seen:
            continue
        seen.add(t)
        tokens.append(t)
        if len(tokens) >= max_tokens:
            break

    # Phrases via chunking
    phrases: List[str] = []
    try:
        pos = pos_tag(raw_tokens)
        tree = chunk(pos)
        seenp = set()
        for subtree in tree.subtrees(lambda t: getattr(t, "label", lambda: None)() in ("NP", "VP", "PP")):
            phrase = " ".join([w for (w, _) in subtree.leaves()]).strip()
            if phrase and phrase not in seenp:
                seenp.add(phrase)
                phrases.append(phrase)
                if len(phrases) >= max_phrases:
                    break
    except Exception:
        # Fallback: bigrams from tokens
        for i in range(len(tokens) - 1):
            ph = f"{tokens[i]} {tokens[i+1]}"
            phrases.append(ph)
            if len(phrases) >= max_phrases:
                break

    return q_norm, phrases, tokens


_TSQUERY_SPECIALS = r"[&|!():']"


def escape_tsquery_term(term: str) -> str:
    """
    Remove tsquery special chars. Spaces are preserved for phrase handling.
    """
    term = re.sub(_TSQUERY_SPECIALS, " ", term)
    term = " ".join(term.split())
    return term


def _lexeme_or_phrase_for_tsquery(term: str) -> str:
    """
    Convert a possibly multi-word term into a tsquery-safe expression:
    - Single token -> token
    - Multi-word -> (tok1 & tok2 & tok3) to avoid tsquery syntax errors
      (We use AND inside a token to represent that phrase loosely.)
    """
    term = escape_tsquery_term(term)
    if not term:
        return ""
    if re.search(r"\s", term):
        parts = [p for p in re.split(r"\s+", term) if p]
        if not parts:
            return ""
        return "(" + " & ".join(parts) + ")"
    return term


def build_tokens_tsquery_string(tokens: List[str]) -> str:
    """
    Build an OR tsquery string robustly handling multi-word tokens:
    Example: ["bàn", "làm việc"] -> "bàn | (làm & việc)"
    """
    items: List[str] = []
    for t in tokens:
        expr = _lexeme_or_phrase_for_tsquery(t)
        if expr:
            items.append(expr)
    return " | ".join(items)


def to_pg_vector_literal(vec: List[float]) -> str:
    """
    Convert Python list[float] to pgvector literal: [x1,x2,...]
    """
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


def build_search_sql(filter_by_ids: bool = False) -> str:
    """
    SQL with parameters (psycopg2 placeholders %s):
      1: full_sentence (text)
      2: phrases[] (text[])
      3: tokens[] (text[])
      4: tokens_tsquery_string (text)  -- guard value
      5: tokens_tsquery_string (text)  -- actual to_tsquery
      6: phrases[] (text[])            -- for EXISTS
      7: tokens[] (text[])             -- for EXISTS
      8: query_vector (vector literal string)
      9: alpha (float8)
     10: alpha (float8)                -- used twice
     11: product_ids (int[])           -- MOVED HERE when filter_by_ids=True
     12: limit_k (int)                 -- MOVED TO END
    """
    product_filter = "AND p.id = ANY(%s::int[])" if filter_by_ids else ""
    return f"""
        WITH q AS (
            SELECT websearch_to_tsquery('simple', %s) AS full_q
        ),
        cand AS (
            SELECT
                p.*,
                (
                    (p.tsv @@ (SELECT full_q FROM q))::int * 10
                    + COALESCE((
                        SELECT SUM( (p.tsv @@ plainto_tsquery('simple', ph))::int )
                        FROM unnest(%s::text[]) AS ph
                    ), 0) * 3
                    + COALESCE((
                        SELECT SUM( (p.tsv @@ plainto_tsquery('simple', tk))::int )
                        FROM unnest(%s::text[]) AS tk
                    ), 0) * 1
                ) AS match_bonus
            FROM products p
            WHERE
                (
                    (p.tsv @@ (SELECT full_q FROM q))
                    OR ( %s <> '' AND p.tsv @@ to_tsquery('simple', %s) )
                    OR EXISTS (
                        SELECT 1 FROM unnest(%s::text[]) ph
                        WHERE p.tsv @@ plainto_tsquery('simple', ph)
                    )
                    OR EXISTS (
                        SELECT 1 FROM unnest(%s::text[]) tk
                        WHERE p.tsv @@ plainto_tsquery('simple', tk)
                    )
                )
                {product_filter}
        ),
        scored AS (
            SELECT
                id,
                title,
                category,
                match_bonus,
                (1 - (embedding <=> %s::vector)) AS vector_score,
                CASE
                    WHEN MAX(match_bonus) OVER () > 0
                        THEN match_bonus::float / MAX(match_bonus) OVER ()
                    ELSE 0.0
                END AS norm_bonus
            FROM cand
        )
        SELECT
            id,
            title,
            category,
            match_bonus,
            vector_score,
            (%s::float8) * vector_score + (1 - %s::float8) * norm_bonus AS total_score
        FROM scored
        ORDER BY total_score DESC
        LIMIT %s
        """


# =========================
# Redis cache (sync)
# =========================

_REDIS: Optional["redis.Redis"] = None  # type: ignore


def _get_redis() -> Optional["redis.Redis"]:  # type: ignore
    global _REDIS
    if redis is None:
        return None
    if _REDIS is None:
        try:
            _REDIS = redis.from_url(REDIS_URL, decode_responses=True)
        except Exception:
            _REDIS = None
    return _REDIS


def _make_cache_key(query: str, page_size: int) -> str:
    """
    Cache key depends on normalized query + page_size + alpha + model tag.
    """
    base = "\n".join(
        [
            f"v={_CACHE_VERSION}",
            f"q={query}",
            f"ps={page_size}",
            f"alpha={ALPHA}",
            f"mtag={CACHE_MODEL_TAG}",
        ]
    )
    h = hashlib.sha256(base.encode("utf-8")).hexdigest()
    return f"search:{h}"


def _cache_set_full_list(key: str, data: List[Dict[str, Any]]) -> None:
    r = _get_redis()
    if r is None:
        return
    try:
        r.setex(key, CACHE_TTL, json.dumps(data, ensure_ascii=False))
    except Exception:
        pass  # ignore cache errors


def _cache_get_full_list(key: str) -> Optional[List[Dict[str, Any]]]:
    r = _get_redis()
    if r is None:
        return None
    try:
        raw = r.get(key)
        if not raw:
            return None
        return json.loads(raw)
    except Exception:
        return None


# =========================
# Core search (sync)
# =========================

def _search_products_core(q: str, page: int, page_size: int, product_ids: Optional[List[int]] = None) -> Dict[str, Any]:
    q_norm = normalize_query(q)
    
    # Debug logging should be here, before any processing
    if product_ids:
        print(f"Filtering by product_ids: {product_ids}")  # Debug log
    else:
        print("No product_ids filter applied.")
    
    # Try cache
    cache_key = _make_cache_key(q_norm, page_size)
    cached = _cache_get_full_list(cache_key)
    if cached is not None:
        total = len(cached)
        start = max(0, (page - 1) * page_size)
        end = min(total, start + page_size)
        page_items = cached[start:end] if start < total else []
        return {"query": q_norm, "page": page, "page_size": page_size, "total": total, "results": page_items}

    # Build query parts
    full_sentence, phrases, tokens = split_query_vi(q_norm)
    tokens_ts = build_tokens_tsquery_string(tokens)
    q_vec = embed_query(full_sentence)  # from your model.py (already normalized)
    q_vec_literal = to_pg_vector_literal([float(x) for x in q_vec])

    sql = build_search_sql(product_ids is not None)
    
    # Build parameters in the correct order
    if product_ids is not None:
        params = [
            full_sentence,           # 1: websearch_to_tsquery
            phrases,                 # 2: phrases for SUM
            tokens,                  # 3: tokens for SUM
            tokens_ts,               # 4: guard <> ''
            tokens_ts,               # 5: to_tsquery
            phrases,                 # 6: phrases for EXISTS
            tokens,                  # 7: tokens for EXISTS
            list(product_ids),       # 8: product_ids filter
            q_vec_literal,           # 9: vector for vector_score
            float(ALPHA),            # 10: alpha
            float(ALPHA),            # 11: alpha again
            int(CANDIDATE_LIMIT),    # 12: limit
        ]
    else:
        params = [
            full_sentence,           # 1: websearch_to_tsquery
            phrases,                 # 2: phrases for SUM
            tokens,                  # 3: tokens for SUM
            tokens_ts,               # 4: guard <> ''
            tokens_ts,               # 5: to_tsquery
            phrases,                 # 6: phrases for EXISTS
            tokens,                  # 7: tokens for EXISTS
            q_vec_literal,           # 8: vector for vector_score
            float(ALPHA),            # 9: alpha
            float(ALPHA),            # 10: alpha again
            int(CANDIDATE_LIMIT),    # 11: limit
        ]

    # Query DB
    conn = get_connection()
    try:
        conn.autocommit = True
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            try:
                cur.execute(sql, params)
            except Exception as e:
                # Safety fallback: if tokens_ts causes tsquery syntax issues,
                # disable the to_tsquery branch and retry with only full/phrase/token EXISTS filters.
                if "tsquery" in str(e).lower():
                    if product_ids is not None:
                        params_fallback = [
                            full_sentence, phrases, tokens,
                            "", "",                 # disable tokens_to_tsquery branch
                            phrases, tokens,
                            q_vec_literal, float(ALPHA), float(ALPHA),
                            list(product_ids),      # product_ids filter
                            int(CANDIDATE_LIMIT),
                        ]
                    else:
                        params_fallback = [
                            full_sentence, phrases, tokens,
                            "", "",                 # disable tokens_to_tsquery branch
                            phrases, tokens,
                            q_vec_literal, float(ALPHA), float(ALPHA),
                            int(CANDIDATE_LIMIT),
                        ]
                    cur.execute(sql, params_fallback)
                else:
                    raise
            rows = cur.fetchall()
    finally:
        try:
            conn.close()
        except Exception:
            pass

    # Rest of the function remains the same...

    # Normalize output types and cache
    full_list: List[Dict[str, Any]] = []
    for r in rows:
        full_list.append(
            {
                "id": int(r["id"]),
                "title": r["title"],
                "category": r["category"],
                "match_bonus": float(r["match_bonus"]),
                "vector_score": float(r["vector_score"]),
                "total_score": float(r["total_score"]),  # already in [0,1]
            }
        )

    _cache_set_full_list(cache_key, full_list)

    total = len(full_list)
    start = max(0, (page - 1) * page_size)
    end = min(total, start + page_size)
    page_items = full_list[start:end] if start < total else []

    return {"query": q_norm, "page": page, "page_size": page_size, "total": total, "results": page_items}

def search_products(q: str, page: int = 1, page_size: int = 20, product_ids: Optional[List[int]] = None) -> Dict[str, Any]:
    """
    Public sync API (callable without FastAPI).
    """
    if not isinstance(q, str) or not q.strip():
        return {"query": "", "page": page, "page_size": page_size, "total": 0, "results": []}
    return _search_products_core(q, page, page_size, product_ids)


# =========================
# FastAPI Router
# =========================

router = APIRouter(prefix="", tags=["search"])  # no extra prefix so endpoint is GET /search

@router.get("/search")
def search_endpoint(
    q: str = Query(..., min_length=1, description="Search query"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    product_ids: Optional[str] = Query(None, description="Comma-separated product IDs")
):
    try:
        # Parse comma-separated product_ids
        parsed_product_ids = None
        if product_ids:
            try:
                parsed_product_ids = [int(pid.strip()) for pid in product_ids.split(',') if pid.strip()]
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid product_ids format. Use comma-separated integers.")
        
        return search_products(q, page=page, page_size=page_size, product_ids=parsed_product_ids)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))