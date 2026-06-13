"""Per-user world state snapshot.

`refresh_world_state(user_id)` reads the live data sources — email_insights,
integration_actions, loops, today's Nylas calendar events, user_matters — and
upserts a compact JSONB snapshot into user_world_state. The background reasoning
agent reads this snapshot instead of querying 6 tables on every reasoning pass.

Trigger points:
  - Nylas email webhook (after classify_and_store_email_insight)
  - Nylas calendar webhook (after any event.created/updated/deleted)
  - Session shutdown callback (post-voice session)
  - Scheduler safety net (every 30 min)
"""
import asyncio
import logging
import os
from datetime import datetime, timezone, timedelta
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_NYLAS_BASE = "https://api.us.nylas.com/v3/grants"


async def refresh_world_state(user_id: str) -> None:
    """Rebuild and upsert the user_world_state snapshot for `user_id`."""
    from .supabase_client import get_supabase_client
    db = get_supabase_client()

    try:
        snapshot = await _build_snapshot(db.client, user_id)
        now_iso = datetime.now(timezone.utc).isoformat()
        await asyncio.to_thread(
            lambda: db.client.table("user_world_state").upsert(
                {"user_id": user_id, "snapshot": snapshot, "updated_at": now_iso},
                on_conflict="user_id",
            ).execute()
        )
        logger.info(f"[world_state] Refreshed for user={user_id}: {snapshot}")
    except Exception as e:
        logger.error(f"[world_state] refresh failed for user={user_id}: {e}", exc_info=True)


async def _build_snapshot(db, user_id: str) -> dict[str, Any]:
    """Aggregate live data into a compact world model dict."""
    urgent_emails, pending_approvals, overdue_tasks, awaiting_replies, todays_meetings = await asyncio.gather(
        _count_urgent_emails(db, user_id),
        _count_pending_approvals(db, user_id),
        _count_overdue_tasks(db, user_id),
        _count_awaiting_replies(db, user_id),
        _get_todays_meetings(db, user_id),
        return_exceptions=True,
    )

    def _safe(val, default):
        return default if isinstance(val, Exception) else val

    return {
        "urgent_emails": _safe(urgent_emails, 0),
        "pending_approvals": _safe(pending_approvals, 0),
        "overdue_tasks": _safe(overdue_tasks, 0),
        "awaiting_replies": _safe(awaiting_replies, 0),
        "todays_meetings": _safe(todays_meetings, []),
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }


async def _count_urgent_emails(db, user_id: str) -> int:
    resp = await asyncio.to_thread(
        lambda: db.table("email_insights")
        .select("id", count="exact")
        .eq("user_id", user_id)
        .eq("insight_type", "needs_attention")
        .eq("is_addressed", False)
        .execute()
    )
    return getattr(resp, "count", None) or len(resp.data or [])


async def _count_pending_approvals(db, user_id: str) -> int:
    resp = await asyncio.to_thread(
        lambda: db.table("integration_actions")
        .select("id", count="exact")
        .eq("user_id", user_id)
        .eq("status", "pending_confirmation")
        .execute()
    )
    return getattr(resp, "count", None) or len(resp.data or [])


async def _count_overdue_tasks(db, user_id: str) -> int:
    now_iso = datetime.now(timezone.utc).isoformat()
    resp = await asyncio.to_thread(
        lambda: db.table("loops")
        .select("id", count="exact")
        .eq("user_id", user_id)
        .neq("status", "done")
        .lt("due_date", now_iso)
        .execute()
    )
    return getattr(resp, "count", None) or len(resp.data or [])


async def _count_awaiting_replies(db, user_id: str) -> int:
    resp = await asyncio.to_thread(
        lambda: db.table("email_insights")
        .select("id", count="exact")
        .eq("user_id", user_id)
        .eq("insight_type", "awaiting_response")
        .eq("is_addressed", False)
        .execute()
    )
    return getattr(resp, "count", None) or len(resp.data or [])


async def _get_todays_meetings(db, user_id: str) -> list[str]:
    """Fetch today's Nylas calendar events for the user. Returns list of title strings."""
    nylas_key = os.getenv("NYLAS_API_KEY", "")
    if not nylas_key:
        return []

    token_resp = await asyncio.to_thread(
        lambda: db.table("nylas_oauth_tokens")
        .select("grant_id")
        .eq("user_id", user_id)
        .eq("integration_type", "calendar")
        .maybe_single()
        .execute()
    )
    if not token_resp or not token_resp.data:
        return []
    grant_id = token_resp.data.get("grant_id")
    if not grant_id:
        return []

    today = datetime.now(timezone.utc).date()
    start_ts = int(datetime(today.year, today.month, today.day, tzinfo=timezone.utc).timestamp())
    end_ts = start_ts + 86400

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(
                f"{_NYLAS_BASE}/{grant_id}/events",
                headers={"Authorization": f"Bearer {nylas_key}", "Accept": "application/json"},
                params={"start": start_ts, "end": end_ts, "limit": 10},
            )
            if resp.status_code != 200:
                return []
            events = resp.json().get("data", [])
            return [e.get("title", "Untitled event") for e in events if e.get("title")]
    except Exception as e:
        logger.warning(f"[world_state] Nylas calendar fetch failed for user={user_id}: {e}")
        return []
