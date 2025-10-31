"""
Hybrid search router for products using:
- Vietnamese query processing (underthesea): full sentence, phrases, tokens
- PostgreSQL FTS (GIN) on Products.TSV
- pgvector cosine similarity on Products.Embedding
- Match bonus: full +10, phrase +3, token +1, noun boosted
- Final score (normalized to [0,1]):
    total_score = ALPHA * vector_score + (1 - ALPHA) * normalized_match_bonus

Filters:
- category_ids: filter by categories
- min_price/max_price: price range filter

Returns: PagedResult<ProductListItemDTO> format for ASP.NET
"""

from __future__ import annotations

import os
import re
import json
import hashlib
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Query, HTTPException
from underthesea import word_tokenize, pos_tag, chunk
from model import embed as embed_query

try:
    from redis import asyncio as redis
except Exception as e:
    print(f"❌ Failed to import redis asyncio: {e}")
    redis = None

from db import get_async_connection

# =========================
# Config
# =========================

ALPHA = float(os.getenv("ALPHA", "0.7"))
CANDIDATE_LIMIT = int(os.getenv("CANDIDATE_LIMIT", "1000"))
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))
CACHE_MODEL_TAG = os.getenv("CACHE_MODEL_TAG", "")

_CACHE_VERSION = "v3_filters"  # bumped for new filter support

# =========================
# Helpers
# =========================

def normalize_query(q: str) -> str:
    return " ".join(q.strip().lower().split())

def strip_accents(s: str) -> str:
    if not s:
        return s
    nkfd = unicodedata.normalize("NFKD", s)
    return "".join([c for c in nkfd if not unicodedata.combining(c)])

def split_query_vi(q: str, max_phrases: int = 10, max_tokens: int = 16) -> Tuple[str, List[str], List[str], List[str]]:
    q_norm = normalize_query(q)

    try:
        raw_tokens = word_tokenize(q_norm)
    except Exception:
        raw_tokens = q_norm.split()

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

    nouns: List[str] = []
    try:
        pos = pos_tag(raw_tokens)
        seen_n = set()
        for w, tag in pos:
            if tag and str(tag).upper().startswith("N"):
                w = w.strip()
                if w and w not in seen_n:
                    seen_n.add(w)
                    nouns.append(w)
                    if len(nouns) >= max_tokens:
                        break
    except Exception:
        if tokens:
            nouns = [tokens[0]]

    phrases: List[str] = []
    try:
        pos_for_chunk = pos_tag(raw_tokens)
        tree = chunk(pos_for_chunk)
        seenp = set()
        for subtree in tree.subtrees(lambda t: getattr(t, "label", lambda: None)() in ("NP", "VP", "PP")):
            phrase = " ".join([w for (w, _) in subtree.leaves()]).strip()
            if phrase and phrase not in seenp:
                seenp.add(phrase)
                phrases.append(phrase)
                if len(phrases) >= max_phrases:
                    break
    except Exception:
        seenp = set()
        max_ngram = 3 if len(tokens) >= 3 else 2
        for n in range(max_ngram, 1, -1):
            for i in range(0, len(tokens) - n + 1):
                ph = " ".join(tokens[i:i + n]).strip()
                if ph and ph not in seenp:
                    seenp.add(ph)
                    phrases.append(ph)
                    if len(phrases) >= max_phrases:
                        break
            if len(phrases) >= max_phrases:
                break

    return q_norm, phrases, tokens, nouns

_TSQUERY_SPECIALS = r"[&|!():']"

def escape_tsquery_term(term: str) -> str:
    term = re.sub(_TSQUERY_SPECIALS, " ", term)
    term = " ".join(term.split())
    return term

def _lexeme_or_phrase_for_tsquery(term: str) -> str:
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
    items: List[str] = []
    for t in tokens:
        expr = _lexeme_or_phrase_for_tsquery(t)
        if expr:
            items.append(expr)
    return " | ".join(items)

def to_pg_vector_literal(vec: List[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"

def build_search_sql(has_category_filter: bool = False, has_price_filter: bool = False) -> str:
    """
    Returns only ProductIds with relevance scores.
    ASP.NET will fetch full product details.
    """
    filters = []
    param_idx = 15
    
    if has_category_filter:
        filters.append(f'AND EXISTS (SELECT 1 FROM "CategoryProduct" cp WHERE cp."ProductsProductId" = p."ProductId" AND cp."CategoriesCategoryId" = ANY(${param_idx}))')
        param_idx += 1
    
    if has_price_filter:
        filters.append(f'AND p."Price" BETWEEN ${param_idx} AND ${param_idx + 1}')
        param_idx += 2
    
    filter_clause = " ".join(filters)
    limit_param = f"${param_idx}"
    
    return f"""
        WITH q AS (
            SELECT websearch_to_tsquery('simple', unaccent($1)) AS full_q
        ),
        cand AS (
            SELECT
                p."ProductId",
                p."Embedding",
                (
                    (p."TSV" @@ (SELECT full_q FROM q))::int * 10
                    + COALESCE((
                        SELECT SUM( (p."TSV" @@ plainto_tsquery('simple', unaccent(unnest_ph)))::int )
                        FROM unnest($2::text[]) AS unnest_ph
                    ), 0) * 3
                    + COALESCE((
                        SELECT SUM( (p."TSV" @@ plainto_tsquery('simple', unaccent(unnest_tk)))::int )
                        FROM unnest($3::text[]) AS unnest_tk
                    ), 0) * 1
                    + COALESCE((
                        SELECT SUM( (p."TSV" @@ plainto_tsquery('simple', unaccent(n)))::int * (($4 - ord + 1) * 10) )
                        FROM unnest($5::text[]) WITH ORDINALITY AS t(n, ord)
                    ), 0)
                ) AS match_bonus
            FROM "Products" p
            WHERE
                (
                    (p."TSV" @@ (SELECT full_q FROM q))
                    OR ( $6 <> '' AND p."TSV" @@ to_tsquery('simple', unaccent($7)) )
                    OR EXISTS (
                        SELECT 1 FROM unnest($8::text[]) ph
                        WHERE p."TSV" @@ plainto_tsquery('simple', unaccent(ph))
                    )
                    OR EXISTS (
                        SELECT 1 FROM unnest($9::text[]) tk
                        WHERE p."TSV" @@ plainto_tsquery('simple', unaccent(tk))
                    )
                )
                AND ( $10 = 0 OR EXISTS (
                    SELECT 1 FROM unnest($11::text[]) nn WHERE p."TSV" @@ plainto_tsquery('simple', unaccent(nn))
                ))
                {filter_clause}
        ),
        scored AS (
            SELECT
                "ProductId",
                match_bonus,
                (1 - ("Embedding" <=> $12::vector)) AS vector_score,
                CASE
                    WHEN MAX(match_bonus) OVER () > 0
                        THEN match_bonus::float / MAX(match_bonus) OVER ()
                    ELSE 0.0
                END AS norm_bonus
            FROM cand
        )
        SELECT
            "ProductId",
            ($13::float8) * vector_score + (1 - $14::float8) * norm_bonus AS relevance_score
        FROM scored
        ORDER BY relevance_score DESC
        LIMIT {limit_param}
    """

# =========================
# Redis cache
# =========================

_REDIS: Optional["redis.Redis"] = None

async def _get_redis() -> Optional["redis.Redis"]:
    global _REDIS
    if redis is None:
        return None
    if _REDIS is None:
        try:
            _REDIS = await redis.from_url(REDIS_URL)
        except Exception:
            _REDIS = None
    return _REDIS

def _make_cache_key(
    query: str, 
    page_size: int, 
    category_ids: Optional[List[int]] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None
) -> str:
    """Cache key includes ALL filters"""
    filter_parts = []
    
    if category_ids:
        sorted_ids = ",".join(str(x) for x in sorted(category_ids))
        filter_parts.append(f"cids={sorted_ids}")
    
    if min_price is not None:
        filter_parts.append(f"minp={min_price}")
    
    if max_price is not None:
        filter_parts.append(f"maxp={max_price}")
    
    base = "\n".join(
        [
            f"v={_CACHE_VERSION}",
            f"q={query}",
            f"ps={page_size}",
            f"alpha={ALPHA}",
            f"mtag={CACHE_MODEL_TAG}",
        ] + filter_parts
    )
    h = hashlib.sha256(base.encode("utf-8")).hexdigest()
    return f"search:{h}"

async def _cache_set_full_list(key: str, data: List[Dict[str, Any]]) -> None:
    r = await _get_redis()
    if r is None:
        return
    try:
        await r.setex(key, CACHE_TTL, json.dumps(data, ensure_ascii=False, default=str))
    except Exception:
        pass

async def _cache_get_full_list(key: str) -> Optional[List[Dict[str, Any]]]:
    r = await _get_redis()
    if r is None:
        return None
    try:
        raw = await r.get(key)
        if not raw:
            return None
        return json.loads(raw)
    except Exception:
        return None

# =========================
# Core search
# =========================

async def _search_products_core(
    q: str, 
    page: int, 
    page_size: int, 
    category_ids: Optional[List[int]] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None
) -> Dict[str, Any]:
    q_norm = normalize_query(q)

    cache_key = _make_cache_key(q_norm, page_size, category_ids, min_price, max_price)

    cached = await _cache_get_full_list(cache_key)
    if cached is not None:
        total = len(cached)
        start = max(0, (page - 1) * page_size)
        end = min(total, start + page_size)
        page_items = cached[start:end] if start < total else []
        return {
            "productIds": page_items,
            "totalCount": total,
            "page": page,
            "pageSize": page_size
        }

    full_sentence, phrases, tokens, nouns = split_query_vi(q_norm)
    tokens_ts = build_tokens_tsquery_string(tokens)
    q_vec = embed_query(full_sentence)
    q_vec_literal = to_pg_vector_literal([float(x) for x in q_vec])

    sql = build_search_sql(
        has_category_filter=category_ids is not None,
        has_price_filter=min_price is not None or max_price is not None
    )

    num_nouns = len(nouns)

    params = [
        full_sentence,  # 1
        phrases,        # 2
        tokens,         # 3
        num_nouns,      # 4
        nouns,          # 5
        tokens_ts,      # 6
        tokens_ts,      # 7
        phrases,        # 8
        tokens,         # 9
        num_nouns,      #10
        nouns,          #11
        q_vec_literal,  #12
        float(ALPHA),   #13
        float(ALPHA),   #14
    ]

    if category_ids:
        params.append(list(category_ids))  # 15
    if min_price is not None and max_price is not None:
        params.append(float(min_price))   # 16
        params.append(float(max_price))   # 17
    
    params.append(int(CANDIDATE_LIMIT))  # final: limit

    conn = await get_async_connection()
    try:
        try:
            rows = await conn.fetch(sql, *params)
        except Exception as e:
            if "tsquery" in str(e).lower():
                params_fallback = params.copy()
                params_fallback[5] = ""  # tokens_ts guard
                params_fallback[6] = ""  # tokens_ts for to_tsquery
                print("tsquery fallback, original error:", str(e))
                rows = await conn.fetch(sql, *params_fallback)
            else:
                raise
    finally:
        await conn.close()

    # Return only ProductIds with optional relevance scores
    product_ids = [int(r["ProductId"]) for r in rows]

    await _cache_set_full_list(cache_key, product_ids)

    total = len(product_ids)
    start = max(0, (page - 1) * page_size)
    end = min(total, start + page_size)
    page_items = product_ids[start:end] if start < total else []

    return {
        "productIds": page_items,
        "totalCount": total,
        "page": page,
        "pageSize": page_size
    }

async def search_products(
    q: str, 
    page: int = 1, 
    page_size: int = 20, 
    category_ids: Optional[List[int]] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None
) -> Dict[str, Any]:
    """Public API"""
    if not isinstance(q, str) or not q.strip():
        return {"items": [], "totalCount": 0, "page": page, "pageSize": page_size}
    return await _search_products_core(q, page, page_size, category_ids, min_price, max_price)

# =========================
# FastAPI Router
# =========================

router = APIRouter(prefix="", tags=["search"])

@router.get("/search")
async def search_endpoint(
    q: str = Query(..., min_length=1, description="Search query"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    category_ids: Optional[str] = Query(None, description="Comma-separated category IDs"),
    min_price: Optional[float] = Query(None, ge=0),
    max_price: Optional[float] = Query(None, ge=0)
):
    try:
        parsed_category_ids = None
        if category_ids:
            try:
                parsed_category_ids = [int(cid.strip()) for cid in category_ids.split(',') if cid.strip()]
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid category_ids format.")
        
        return await search_products(
            q, 
            page=page, 
            page_size=page_size, 
            category_ids=parsed_category_ids,
            min_price=min_price,
            max_price=max_price
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
