"""
Hybrid search router for products (title + category) using:
- Vietnamese query processing (underthesea): full sentence, phrases, tokens
- PostgreSQL FTS (GIN) on products.tsv
- pgvector cosine similarity on products.embedding
- Match bonus: full +10, phrase +3, token +1, noun boosted
- Final score (normalized to [0,1]):
    total_score = ALPHA * vector_score + (1 - ALPHA) * normalized_match_bonus
  where:
    - vector_score ∈ [0,1] = 1 - (embedding <=> query_vec)
    - normalized_match_bonus ∈ [0,1] = match_bonus / max(match_bonus) over this query's candidates

This version:
- Uses synchronous psycopg2 connection from db.get_connection()
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
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

# FastAPI router
from fastapi import APIRouter, Query, HTTPException

# Vietnamese NLP
from underthesea import word_tokenize, pos_tag, chunk
# (optionally) pyvi can be used if you prefer: from pyvi import ViTokenizer

# Your existing embedder
from model import embed as embed_query  # type: ignore

# Redis (sync) is optional
try:
    from redis import asyncio as redis  # ✅ modern redis client (replaces aioredis)
except Exception as e:
    print(f"❌ Failed to import redis asyncio: {e}")
    redis = None  # type: ignore

# DB connection (psycopg2)
from db import get_async_connection  # type: ignore
# from psycopg2.extras import RealDictCursor  # type: ignore


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


def strip_accents(s: str) -> str:
    """Remove diacritics (for fallback token generation if DB unaccent not used)."""
    if not s:
        return s
    nkfd = unicodedata.normalize("NFKD", s)
    return "".join([c for c in nkfd if not unicodedata.combining(c)])


def split_query_vi(q: str, max_phrases: int = 10, max_tokens: int = 16) -> Tuple[str, List[str], List[str], List[str]]:
    """
    Returns (full_sentence, phrases, tokens, nouns)
    - tokens: unique, order-preserving, truncated to max_tokens
    - phrases: try chunking if possible, fallback to contiguous ngrams (tri/bi)
    - nouns: tokens detected as nouns (order-preserving), truncated to max_tokens
    """
    q_norm = normalize_query(q)

    # Basic tokenization
    try:
        raw_tokens = word_tokenize(q_norm)
    except Exception:
        # fallback to simple split
        raw_tokens = q_norm.split()

    # Build tokens (unique, ordered)
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

    # Attempt POS tagging to extract nouns
    nouns: List[str] = []
    try:
        pos = pos_tag(raw_tokens)
        seen_n = set()
        for w, tag in pos:
            # underthesea tags nouns often with prefix "N" (N, Np, Nc, Nu, etc.)
            if tag and str(tag).upper().startswith("N"):
                w = w.strip()
                if w and w not in seen_n:
                    seen_n.add(w)
                    nouns.append(w)
                    if len(nouns) >= max_tokens:
                        break
    except Exception:
        # fallback heuristic: take first token as noun candidate
        if tokens:
            nouns = [tokens[0]]

    # Phrases: try to use chunk() if available, fallback to n-grams
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
        # fallback: contiguous n-grams (prefer trigrams then bigrams)
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
    SQL placeholders order (psycopg2 %s) in this SQL:
    1: full_sentence (websearch_to_tsquery/unaccent)
    2: phrases[] for SUM (unnest_ph)
    3: tokens[] for SUM (unnest_tk)
    4: num_nouns (used inside noun bonus multiplier)
    5: nouns[] (for noun bonus)
    6: tokens_ts_string (guard)
    7: tokens_ts_string (to_tsquery)
    8: phrases[] (for EXISTS)
    9: tokens[] (for EXISTS)
    10: num_nouns (for AND check)
    11: nouns[] (for AND check)
    12: query_vector (vector literal)
    13: alpha
    14: alpha    -- used twice
    15: product_ids (int[]) -- if filter_by_ids True (appears before LIMIT)
    16: limit_k   -- when filter_by_ids True, otherwise 15 is limit_k
    """
    product_filter = "AND p.id = ANY($15)" if filter_by_ids else ""
    return f"""
        WITH q AS (
            -- full sentence is unaccented before building websearch_to_tsquery
            SELECT websearch_to_tsquery('simple', unaccent($1)) AS full_q
        ),
        cand AS (
            SELECT
                p.*,
                (
                    (p.tsv @@ (SELECT full_q FROM q))::int * 10
                    + COALESCE((
                        -- phrases SUM: unaccent each phrase parameter before plainto_tsquery
                        SELECT SUM( (p.tsv @@ plainto_tsquery('simple', unaccent(unnest_ph)))::int )
                        FROM unnest($2::text[]) AS unnest_ph
                    ), 0) * 3
                    + COALESCE((
                        -- tokens SUM: unaccent each token parameter before plainto_tsquery
                        SELECT SUM( (p.tsv @@ plainto_tsquery('simple', unaccent(unnest_tk)))::int )
                        FROM unnest($3::text[]) AS unnest_tk
                    ), 0) * 1
                    + COALESCE((
                        -- noun bonus: unaccent nouns and use ordinality for position-based multiplier
                        SELECT SUM( (p.tsv @@ plainto_tsquery('simple', unaccent(n)))::int * (($4 - ord + 1) * 10) )
                        FROM unnest($5::text[]) WITH ORDINALITY AS t(n, ord)
                    ), 0)
                ) AS match_bonus
            FROM products p
            WHERE
                (
                    (p.tsv @@ (SELECT full_q FROM q))
                    -- tokens_ts guard: apply unaccent to the tokens_ts string before to_tsquery
                    OR ( $6 <> '' AND p.tsv @@ to_tsquery('simple', unaccent($7)) )
                    OR EXISTS (
                        -- phrases existence: unaccent phrase element
                        SELECT 1 FROM unnest($8::text[]) ph
                        WHERE p.tsv @@ plainto_tsquery('simple', unaccent(ph))
                    )
                    OR EXISTS (
                        -- tokens existence: unaccent token element
                        SELECT 1 FROM unnest($9::text[]) tk
                        WHERE p.tsv @@ plainto_tsquery('simple', unaccent(tk))
                    )
                )
                -- if num_nouns > 0 then require at least one noun match (unaccented)
                AND ( $10 = 0 OR EXISTS (
                    SELECT 1 FROM unnest($11::text[]) nn WHERE p.tsv @@ plainto_tsquery('simple', unaccent(nn))
                ))
                {product_filter}
        ),
        scored AS (
            SELECT
                id,
                title,
                category,
                match_bonus,
                (1 - (embedding <=> $12::vector)) AS vector_score,
                CASE
                    WHEN MAX(match_bonus) OVER () > 0
                        THEN match_bonus::float / MAX(match_bonus) OVER ()
                    ELSE 0.0
                END AS norm_bonus
            FROM cand
        )
        SELECT
            id
        FROM scored
        ORDER BY ($13::float8) * vector_score + (1 - $14::float8) * norm_bonus DESC
        LIMIT $""" + ("16" if filter_by_ids else "15") + """
        """


# =========================
# Redis cache (sync)
# =========================

_REDIS: Optional["redis.Redis"] = None  # type: ignore


async def _get_redis() -> Optional["redis.Redis"]:  # type: ignore
    global _REDIS
    if redis is None:
        return None
    if _REDIS is None:
        try:
            _REDIS = await redis.from_url(REDIS_URL)
        except Exception as e:
            _REDIS = None
    return _REDIS


def _make_cache_key(query: str, page_size: int, product_ids: Optional[List[int]] = None) -> str:
    """
    Cache key depends on normalized query + page_size + alpha + model tag + optionally product_ids.
    """
    pid_part = ""
    if product_ids:
        sorted_ids = ",".join(str(x) for x in sorted(product_ids))
        pid_part = f"pids={sorted_ids}"
    base = "\n".join(
        [
            f"v={_CACHE_VERSION}",
            f"q={query}",
            f"ps={page_size}",
            f"alpha={ALPHA}",
            f"mtag={CACHE_MODEL_TAG}",
            pid_part,
        ]
    )
    h = hashlib.sha256(base.encode("utf-8")).hexdigest()
    return f"search:{h}"


async def _cache_set_full_list(key: str, data: List[int]) -> None:
    r = await _get_redis()
    if r is None:
        return
    try:
        await r.setex(key, CACHE_TTL, json.dumps(data, ensure_ascii=False))
    except Exception:
        pass  # ignore cache errors


async def _cache_get_full_list(key: str) -> Optional[List[int]]:
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
# Core search (sync)
# =========================

async def _search_products_core(q: str, page: int, page_size: int, product_ids: Optional[List[int]] = None) -> Dict[str, Any]:
    q_norm = normalize_query(q)

    if product_ids:
        print(f"Filtering by product_ids: {product_ids}")
    else:
        print("No product_ids filter applied.")

    # Make cache key with product_ids considered
    cache_key = _make_cache_key(q_norm, page_size, product_ids)

    cached = await _cache_get_full_list(cache_key)
    if cached is not None:
        total = len(cached)
        start = max(0, (page - 1) * page_size)
        end = min(total, start + page_size)
        page_items = cached[start:end] if start < total else []
        return {"query": q_norm, "page": page, "page_size": page_size, "total": total, "results": page_items}
    # Build query parts (now returns nouns as well)
    full_sentence, phrases, tokens, nouns = split_query_vi(q_norm)
    tokens_ts = build_tokens_tsquery_string(tokens)
    # If you want extra robustness for unaccented clients but DB is not unaccented,
    # you could append unaccented tokens into tokens/phrases here. Better: enable DB unaccent.
    q_vec = embed_query(full_sentence)  # from model.py (already normalized)
    q_vec_literal = to_pg_vector_literal([float(x) for x in q_vec])

    sql = build_search_sql(product_ids is not None)

    num_nouns = len(nouns)

    # Build params in the exact order expected by build_search_sql
    if product_ids is not None:
        params = [
            full_sentence,  # 1: websearch_to_tsquery (unaccent inside SQL)
            phrases,        # 2: unnest_ph for SUM
            tokens,         # 3: unnest_tk for SUM
            num_nouns,      # 4: used in noun bonus multiplier (%s - ord + 1)
            nouns,          # 5: nouns[] for noun bonus
            tokens_ts,      # 6: tokens_ts guard
            tokens_ts,      # 7: tokens_ts for to_tsquery
            phrases,        # 8: phrases[] for EXISTS
            tokens,         # 9: tokens[] for EXISTS
            num_nouns,      #10: num_nouns for AND check
            nouns,          #11: nouns[] for AND check
            q_vec_literal,  #12: vector literal
            float(ALPHA),   #13: alpha
            float(ALPHA),   #14: alpha
            list(product_ids),  #15: product_ids filter
            int(CANDIDATE_LIMIT),#16: limit
        ]
    else:
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
            int(CANDIDATE_LIMIT), #15 (limit when no product_ids)
        ]

    # Defensive check: ensure params count matches placeholders (useful for dev)
    expected_placeholders = sql.count("$")
    if len(params) != expected_placeholders:
        # helpful debug info before raising
        raise RuntimeError(f"Param/placeholder mismatch: expected {expected_placeholders} params but got {len(params)}. "
                           f"Query: {full_sentence} tokens_ts='{tokens_ts}' num_nouns={num_nouns} "
                           f"phrases_len={len(phrases)} tokens_len={len(tokens)} nouns_len={len(nouns)}")

    # Query DB
    conn = await get_async_connection()
    try:
            try:
                rows = await conn.fetch(sql, *params)
            except Exception as e:
                # Safety fallback: if tokens_ts causes tsquery syntax issues, retry with tokens_ts disabled
                if "tsquery" in str(e).lower():
                    # rebuild fallback params with tokens_ts disabled (keep same length/order)
                    if product_ids is not None:
                        params_fallback = [
                            full_sentence,
                            phrases,
                            tokens,
                            num_nouns,
                            nouns,
                            "",      # tokens_ts guard disabled
                            "",      # tokens_ts for to_tsquery disabled
                            phrases,
                            tokens,
                            num_nouns,
                            nouns,
                            q_vec_literal,
                            float(ALPHA),
                            float(ALPHA),
                            list(product_ids),
                            int(CANDIDATE_LIMIT),
                        ]
                    else:
                        params_fallback = [
                            full_sentence,
                            phrases,
                            tokens,
                            num_nouns,
                            nouns,
                            "", 
                            "",    # disable tokens_ts parts
                            phrases,
                            tokens,
                            num_nouns,
                            nouns,
                            q_vec_literal,
                            float(ALPHA),
                            float(ALPHA),
                            int(CANDIDATE_LIMIT),
                        ]

                    # defensive check for fallback consistency
                    if len(params_fallback) != expected_placeholders:
                        raise RuntimeError(f"Fallback param/placeholder mismatch: expected {expected_placeholders} but got {len(params_fallback)}. "
                                           f"Original error: {str(e)}")

                    # optional: log original exception e for debugging
                    print("tsquery fallback, original error:", str(e))
                    rows = await conn.fetch(sql, *params_fallback)
                else:
                    raise
    finally:
        await conn.close()

    # Normalize output types and cache
    full_list = [int(r["id"]) for r in rows]
    await _cache_set_full_list(cache_key, full_list)
    print(f"💾 Stored {len(full_list)} results to cache")
    verification = await _cache_get_full_list(cache_key)
    if verification is not None:
        print(f"✅ Verification: Cache now contains {len(verification)} items")
    else:
        print("❌ Verification failed: Cache is still empty after storage!")

    total = len(full_list)
    start = max(0, (page - 1) * page_size)
    end = min(total, start + page_size)
    page_items = full_list[start:end] if start < total else []

    return {"query": q_norm, "page": page, "page_size": page_size, "total": total, "results": page_items}


async def search_products(q: str, page: int = 1, page_size: int = 20, product_ids: Optional[List[int]] = None) -> Dict[str, Any]:
    """
    Public sync API (callable without FastAPI).
    """
    if not isinstance(q, str) or not q.strip():
        return {"query": "", "page": page, "page_size": page_size, "total": 0, "results": []}
    return await _search_products_core(q, page, page_size, product_ids)


# =========================
# FastAPI Router
# =========================

router = APIRouter(prefix="", tags=["search"])  # no extra prefix so endpoint is GET /search


@router.get("/search")
async def search_endpoint(
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
        return await search_products(q, page=page, page_size=page_size, product_ids=parsed_product_ids)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
