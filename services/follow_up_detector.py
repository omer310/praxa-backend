"""Follow-up detector: surface overdue outreach and stale high-priority tasks.

Runs daily (after relationship_linker) via the scheduler. For each active user:
  1. Checks email threads where the user sent the last message more than N days
     ago and has received no reply — surfaces these as nudges.
  2. Checks tasks due within 48 hours that have not been touched recently.

All alerts are delivered via notify_service so delivery preferences and quiet
hours are respected automatically.
"""
import asyncio
import logging
import os
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

_FOLLOW_UP_DAYS = int(os.getenv("FOLLOW_UP_DAYS", "3"))
_TASK_DEADLINE_HOURS = int(os.getenv("TASK_DEADLINE_HOURS", "48"))


def _enabled() -> bool:
    from .proactive import flag_enabled
    return flag_enabled("FOLLOW_UP_DETECTOR_ENABLED", default=True)


async def _active_user_ids() -> list[str]:
    from .supabase_client import get_supabase_client
    db = get_supabase_client()
    try:
        resp = await asyncio.to_thread(
            lambda: db.client.table("user_settings")
            .select("user_id")
            .execute()
        )
        return [r["user_id"] for r in (resp.data or []) if r.get("user_id")]
    except Exception as e:
        logger.error(f"[follow_up] active users fetch failed: {e}")
        return []


async def _check_unanswered_threads(user_id: str, days: int) -> list[dict]:
    """Return email threads where the user sent the last message and got no reply."""
    from .supabase_client import get_supabase_client
    db = get_supabase_client()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    try:
        resp = await asyncio.to_thread(
            lambda: db.client.table("email_insights")
            .select("email_id, from_name, from_email, subject, created_at")
            .eq("user_id", user_id)
            .eq("insight_type", "awaiting_response")
            .eq("is_addressed", False)
            .lte("created_at", cutoff)
            .order("created_at", desc=True)
            .limit(5)
            .execute()
        )
        return resp.data or []
    except Exception as e:
        logger.warning(f"[follow_up] thread check failed for {user_id}: {e}")
        return []


async def _check_due_soon_tasks(user_id: str, hours: int) -> list[dict]:
    """Return tasks due within `hours` hours that haven't been touched recently."""
    from .supabase_client import get_supabase_client
    db = get_supabase_client()
    now = datetime.now(timezone.utc)
    deadline = (now + timedelta(hours=hours)).date().isoformat()
    stale_cutoff = (now - timedelta(hours=12)).isoformat()
    try:
        resp = await asyncio.to_thread(
            lambda: db.client.table("loops")
            .select("id, title, due_date, priority, updated_at")
            .eq("user_id", user_id)
            .lte("due_date", deadline)
            .neq("status", "done")
            .eq("archived", False)
            .lte("updated_at", stale_cutoff)
            .order("due_date")
            .limit(5)
            .execute()
        )
        return resp.data or []
    except Exception as e:
        logger.warning(f"[follow_up] task due check failed for {user_id}: {e}")
        return []


async def run_follow_up_for_user(user_id: str) -> int:
    """Run follow-up detection for a single user. Returns number of nudges sent."""
    from .notify_service import notify_user
    from .proactive import is_allowed

    if not is_allowed(user_id):
        return 0

    nudges_sent = 0

    threads = await _check_unanswered_threads(user_id, _FOLLOW_UP_DAYS)
    if threads:
        names = [t.get("from_name") or t.get("from_email") or "someone" for t in threads[:3]]
        who = ", ".join(names)
        body = (
            f"Still waiting on replies from: {who}. "
            f"It's been over {_FOLLOW_UP_DAYS} days — want to follow up?"
        )
        await notify_user(
            user_id=user_id,
            event_type="follow_up_needed",
            title="Follow-up needed",
            body=body,
            data={"route": "/email-mode", "count": len(threads)},
        )
        nudges_sent += 1
        logger.info(f"[follow_up] sent thread nudge for {user_id}: {len(threads)} threads")

    tasks = await _check_due_soon_tasks(user_id, _TASK_DEADLINE_HOURS)
    if tasks:
        task_names = [t.get("title", "a task") for t in tasks[:3]]
        label = ", ".join(f"'{n}'" for n in task_names)
        body = (
            f"{len(tasks)} task{'s' if len(tasks) > 1 else ''} due within "
            f"{_TASK_DEADLINE_HOURS} hours: {label}."
        )
        await notify_user(
            user_id=user_id,
            event_type="task_due",
            title="Upcoming deadlines",
            body=body,
            data={"route": "/(tabs)/initiatives", "count": len(tasks)},
        )
        nudges_sent += 1
        logger.info(f"[follow_up] sent task nudge for {user_id}: {len(tasks)} tasks due soon")

    return nudges_sent


async def run_follow_up_detector() -> int:
    """Entry point for the scheduler. Returns total nudges sent across all users."""
    if not _enabled():
        return 0

    user_ids = await _active_user_ids()
    total = 0
    for user_id in user_ids:
        try:
            total += await run_follow_up_for_user(user_id)
        except Exception as e:
            logger.error(f"[follow_up] failed for {user_id}: {e}", exc_info=True)

    if total:
        logger.info(f"[follow_up] sent {total} nudge(s) across {len(user_ids)} user(s)")
    return total
