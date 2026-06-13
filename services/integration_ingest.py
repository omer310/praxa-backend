"""Deep ingestion enrichment for integration_context.

The n8n notion-sync workflow performs the lightweight metadata pass (titles,
URLs, ids). This backend job enriches those rows with the heavier signal that
powers semantic retrieval:

  - fetches block content from the provider (Notion) using the user's token,
  - stores a bounded plaintext snippet,
  - embeds title+content with text-embedding-3-small (1536-d),
  - sets expires_at so stale context is purged (privacy: minimal persistence).

Kept in Python (not n8n) because per-row content fetch + embedding + bounded
snippets are far easier to get right and test here than in a fan-out workflow.
"""
import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone

import httpx
from openai import AsyncOpenAI

from .supabase_client import get_supabase_client

logger = logging.getLogger(__name__)

_CONTENT_TTL_DAYS = int(os.getenv("INTEGRATION_CONTEXT_TTL_DAYS", "30"))
_SNIPPET_MAX = 4000
_BATCH = 25
_NOTION_VERSION = "2022-06-28"


def _openai() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _vector_literal(embedding: list[float]) -> str:
    """pgvector expects a bracketed string literal over PostgREST."""
    return "[" + ",".join(f"{x:.6f}" for x in embedding) + "]"


async def _fetch_token(user_id: str, provider: str) -> str | None:
    db = get_supabase_client()
    try:
        resp = await asyncio.to_thread(
            lambda: db.client.table("user_integrations")
            .select("access_token")
            .eq("user_id", user_id)
            .eq("provider", provider)
            .eq("status", "connected")
            .maybe_single()
            .execute()
        )
        if resp and resp.data:
            return resp.data.get("access_token")
    except Exception as e:
        logger.error(f"[ingest] token fetch failed ({provider}) for {user_id}: {e}")
    return None


def _rich_text_to_plain(rich: list) -> str:
    return "".join(rt.get("plain_text", "") for rt in (rich or []))


def _block_to_text(block: dict) -> str:
    btype = block.get("type", "")
    data = block.get(btype, {})
    if isinstance(data, dict) and "rich_text" in data:
        return _rich_text_to_plain(data.get("rich_text", []))
    return ""


async def _fetch_notion_content(client: httpx.AsyncClient, token: str, page_id: str) -> str:
    try:
        resp = await client.get(
            f"https://api.notion.com/v1/blocks/{page_id}/children",
            params={"page_size": 50},
            headers={
                "Authorization": f"Bearer {token}",
                "Notion-Version": _NOTION_VERSION,
            },
        )
        if resp.status_code != 200:
            return ""
        blocks = resp.json().get("results", [])
        lines = [t for b in blocks if (t := _block_to_text(b).strip())]
        return "\n".join(lines)[:_SNIPPET_MAX]
    except Exception as e:
        logger.warning(f"[ingest] notion content fetch failed for {page_id}: {e}")
        return ""


async def _upsert_notion_matter(db, user_id: str, external_id: str | None, title: str | None, content: str) -> None:
    """Best-effort: track an enriched Notion page as an open matter (deduped by source)."""
    if not (user_id and external_id and title):
        return
    try:
        await asyncio.to_thread(
            lambda: db.client.rpc("upsert_matter", {
                "p_user_id": user_id,
                "p_title": title[:160],
                "p_source_type": "notion",
                "p_source_id": external_id,
                "p_description": (content or "")[:500] or None,
            }).execute()
        )
    except Exception as e:
        logger.warning(f"[ingest] upsert_matter failed for {external_id}: {e}")


async def enrich_integration_context(limit: int = _BATCH) -> int:
    """Enrich up to `limit` un-embedded Notion rows. Returns rows updated."""
    db = get_supabase_client()
    try:
        resp = await asyncio.to_thread(
            lambda: db.client.table("integration_context")
            .select("id, user_id, external_id, title, content_type")
            .eq("provider", "notion")
            .eq("content_type", "notion_page")
            .is_("embedding", "null")
            .limit(limit)
            .execute()
        )
        rows = resp.data or []
    except Exception as e:
        logger.error(f"[ingest] query failed: {e}")
        return 0

    if not rows:
        return 0

    openai = _openai()
    token_cache: dict[str, str | None] = {}
    updated = 0
    expires = (_now() + timedelta(days=_CONTENT_TTL_DAYS)).isoformat()

    async with httpx.AsyncClient(timeout=20.0) as client:
        for row in rows:
            user_id = row["user_id"]
            if user_id not in token_cache:
                token_cache[user_id] = await _fetch_token(user_id, "notion")
            token = token_cache[user_id]
            if not token:
                continue

            content = await _fetch_notion_content(client, token, row["external_id"])
            embed_input = f"{row.get('title') or ''}\n{content}".strip()
            if not embed_input:
                continue

            try:
                emb_resp = await openai.embeddings.create(
                    model="text-embedding-3-small",
                    input=embed_input[:8000],
                )
                embedding = emb_resp.data[0].embedding
            except Exception as e:
                logger.warning(f"[ingest] embedding failed for {row['id']}: {e}")
                continue

            try:
                await asyncio.to_thread(
                    lambda r=row, c=content, emb=embedding: db.client.table("integration_context")
                    .update({
                        "content": c,
                        "embedding": _vector_literal(emb),
                        "expires_at": expires,
                        "content_encrypted": False,
                        "last_synced_at": _now().isoformat(),
                    })
                    .eq("id", r["id"])
                    .execute()
                )
                updated += 1
                await _upsert_notion_matter(db, user_id, row.get("external_id"), row.get("title"), content)
            except Exception as e:
                logger.error(f"[ingest] update failed for {row['id']}: {e}")

    if updated:
        logger.info(f"[ingest] enriched {updated} integration_context row(s)")
    return updated
