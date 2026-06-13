"""Relationship graph maintenance (daily).

For each user with a relationship graph:
  - refresh reciprocity-based VIP status (`refresh_contact_vip`),
  - embed open matters that lack an embedding (so `match_matters` works),
  - best-effort link `email_contacts` to a Notion page by name match.

All steps are best-effort and isolated so one failure never blocks the rest.
Kill-switch `RELATIONSHIP_LINKING_ENABLED` (default on); honors PROACTIVE_ALLOWLIST.
"""
import asyncio
import logging

from .proactive import flag_enabled, is_allowed
from .supabase_client import get_supabase_client

logger = logging.getLogger(__name__)

_MATTER_EMBED_LIMIT = 50
_CONTACT_LINK_LIMIT = 100


def _enabled() -> bool:
    return flag_enabled("RELATIONSHIP_LINKING_ENABLED", default=True)


def _vector_literal(embedding: list[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in embedding) + "]"


async def _graph_user_ids() -> list[str]:
    db = get_supabase_client()
    try:
        resp = await asyncio.to_thread(
            lambda: db.client.table("email_contacts").select("user_id").limit(10000).execute()
        )
        return list({r["user_id"] for r in (resp.data or []) if r.get("user_id")})
    except Exception as e:
        logger.error(f"[linker] user fetch failed: {e}")
        return []


async def _refresh_vip(user_id: str) -> None:
    db = get_supabase_client()
    try:
        await asyncio.to_thread(
            lambda: db.client.rpc("refresh_contact_vip", {"p_user_id": user_id}).execute()
        )
    except Exception as e:
        logger.warning(f"[linker] refresh_contact_vip failed for {user_id}: {e}")


async def _embed_matters(user_id: str) -> int:
    from praxa_core.memory import embed_text

    db = get_supabase_client()
    try:
        resp = await asyncio.to_thread(
            lambda: db.client.table("user_matters")
            .select("id, title, description")
            .eq("user_id", user_id)
            .is_("embedding", "null")
            .limit(_MATTER_EMBED_LIMIT)
            .execute()
        )
        rows = resp.data or []
    except Exception as e:
        logger.warning(f"[linker] matters fetch failed for {user_id}: {e}")
        return 0

    embedded = 0
    for m in rows:
        text = f"{m.get('title') or ''}\n{m.get('description') or ''}".strip()
        if not text:
            continue
        emb = await embed_text(text)
        if not emb:
            continue
        try:
            await asyncio.to_thread(
                lambda mid=m["id"], e=emb: db.client.table("user_matters")
                .update({"embedding": _vector_literal(e)}).eq("id", mid).execute()
            )
            embedded += 1
        except Exception as e:
            logger.warning(f"[linker] matter embed update failed: {e}")
    return embedded


async def _link_notion_contacts(user_id: str) -> int:
    db = get_supabase_client()
    try:
        contacts_resp = await asyncio.to_thread(
            lambda: db.client.table("email_contacts")
            .select("id, name")
            .eq("user_id", user_id)
            .is_("linked_notion_page_id", "null")
            .eq("archived", False)
            .not_.is_("name", "null")
            .limit(_CONTACT_LINK_LIMIT)
            .execute()
        )
        contacts = [c for c in (contacts_resp.data or []) if (c.get("name") or "").strip()]
        if not contacts:
            return 0
        pages_resp = await asyncio.to_thread(
            lambda: db.client.table("integration_context")
            .select("external_id, title")
            .eq("user_id", user_id)
            .eq("provider", "notion")
            .limit(1000)
            .execute()
        )
        pages = [(p.get("external_id"), (p.get("title") or "").lower()) for p in (pages_resp.data or []) if p.get("external_id") and p.get("title")]
    except Exception as e:
        logger.warning(f"[linker] notion-link fetch failed for {user_id}: {e}")
        return 0

    if not pages:
        return 0

    linked = 0
    for contact in contacts:
        name = (contact.get("name") or "").strip().lower()
        if len(name) < 3:
            continue
        match = next((ext for ext, title in pages if name in title), None)
        if not match:
            continue
        try:
            await asyncio.to_thread(
                lambda cid=contact["id"], ext=match: db.client.table("email_contacts")
                .update({"linked_notion_page_id": ext}).eq("id", cid).execute()
            )
            linked += 1
        except Exception as e:
            logger.warning(f"[linker] contact link update failed: {e}")
    return linked


async def run_relationship_linker() -> dict:
    """Scheduled entry point. Returns aggregate counts."""
    if not _enabled():
        return {"users": 0}

    user_ids = await _graph_user_ids()
    totals = {"users": 0, "matters_embedded": 0, "contacts_linked": 0}
    for user_id in user_ids:
        if not is_allowed(user_id):
            continue
        totals["users"] += 1
        await _refresh_vip(user_id)
        try:
            totals["matters_embedded"] += await _embed_matters(user_id)
        except Exception as e:
            logger.error(f"[linker] embed matters failed for {user_id}: {e}")
        try:
            totals["contacts_linked"] += await _link_notion_contacts(user_id)
        except Exception as e:
            logger.error(f"[linker] link contacts failed for {user_id}: {e}")

    logger.info(f"[linker] done: {totals}")
    return totals
