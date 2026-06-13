"""Unified action dispatcher - the single execution chokepoint for the
`integration_actions` queue.

Reads are direct everywhere; *writes* land in `integration_actions` (queued by
any surface via praxa_core.queue_action). This dispatcher:

  - refuses to execute any row that still needs confirmation
    (`requires_confirmation = true AND confirmed_at IS NULL`),
  - routes Notion/Slack actions to the n8n action-executor webhook,
  - executes tasks/email/calendar natively via praxa_core,
  - writes status/result/error_message/attempts back to the row.

It is driven by an APScheduler poll (safety net for cross-process enqueues from
the voice agent) and can be invoked directly as a FastAPI BackgroundTask after
an in-process enqueue (SMS agent).
"""
import asyncio
import logging
import os
from datetime import datetime, timezone

import httpx

from .supabase_client import get_supabase_client
from .push_service import send_push_notification, get_user_push_token, schedule_receipt_check

logger = logging.getLogger(__name__)

N8N_ACTION_WEBHOOK_URL = os.getenv("N8N_ACTION_WEBHOOK_URL", "")
PRAXA_WEBHOOK_SECRET = os.getenv("PRAXA_WEBHOOK_SECRET", "")

_N8N_PROVIDERS = {"notion", "slack"}
_TERMINAL_STATUSES = {"done", "failed", "cancelled"}
_MAX_ATTEMPTS = 3


# ---------------------------------------------------------------------------
# ToolContext construction (service-role, per action)
# ---------------------------------------------------------------------------

async def _build_tool_context(user_id: str):
    """Build a praxa_core ToolContext for native execution with a service-role client."""
    from praxa_core import ToolContext

    db = get_supabase_client()
    timezone_str = "UTC"
    try:
        resp = await asyncio.to_thread(
            lambda: db.client.table("user_settings").select("timezone").eq("user_id", user_id).maybe_single().execute()
        )
        if resp and resp.data and resp.data.get("timezone"):
            timezone_str = resp.data["timezone"]
    except Exception:
        pass

    return ToolContext(
        user_id=user_id,
        supabase=db.client,
        timezone=timezone_str,
        surface="dispatcher",
        nylas_api_key=os.getenv("NYLAS_API_KEY"),
    )


# ---------------------------------------------------------------------------
# Native handlers (tasks / email / calendar)
# ---------------------------------------------------------------------------

async def _native_reply_to_email(ctx, payload: dict) -> str:
    from praxa_core.tools import email as _email
    return await _email.reply_to_email_impl(
        ctx, payload.get("email_id", ""), payload.get("reply_body", ""), direct=True
    )


async def _native_reschedule_event(ctx, payload: dict) -> str:
    from praxa_core.tools import calendar as _calendar
    return await _calendar.reschedule_calendar_event_impl(
        ctx, payload.get("event_name", ""), payload.get("new_date_time", "")
    )


async def _native_create_task(ctx, payload: dict) -> str:
    from praxa_core.tools import tasks as _tasks
    return await _tasks.create_task_impl(
        ctx, payload.get("title", ""), payload.get("bucket_name", ""),
        payload.get("priority", "medium"), payload.get("due_date"),
    )


async def _native_complete_task(ctx, payload: dict) -> str:
    from praxa_core.tools import tasks as _tasks
    return await _tasks.complete_task_impl(ctx, payload.get("task_title", ""))


NATIVE_HANDLERS = {
    "reply_to_email": _native_reply_to_email,
    "send_email": _native_reply_to_email,
    "reschedule_calendar_event": _native_reschedule_event,
    "create_task": _native_create_task,
    "complete_task": _native_complete_task,
}


# ---------------------------------------------------------------------------
# n8n routing
# ---------------------------------------------------------------------------

async def _dispatch_to_n8n(action: dict) -> tuple[bool, dict | None, str | None]:
    """POST the action to the n8n action-executor. n8n writes the final
    status/result back to the row itself, so we just confirm the hand-off."""
    if not N8N_ACTION_WEBHOOK_URL:
        return False, None, "N8N_ACTION_WEBHOOK_URL not configured"
    body = {
        "action_id": action["id"],
        "user_id": action["user_id"],
        "provider": action["provider"],
        "action_type": action["action_type"],
        "payload": action.get("payload") or {},
        "secret": PRAXA_WEBHOOK_SECRET,
    }
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                N8N_ACTION_WEBHOOK_URL,
                json=body,
                headers={"x-praxa-secret": PRAXA_WEBHOOK_SECRET},
            )
            if resp.status_code in (200, 201):
                return True, {"handed_off": True}, None
            return False, None, f"n8n returned {resp.status_code}: {resp.text[:200]}"
    except Exception as e:
        return False, None, f"n8n request failed: {e}"


# ---------------------------------------------------------------------------
# Core dispatch
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _claim_action(action_id: str, attempts: int) -> bool:
    """Atomically move a row to 'running' so concurrent pollers don't double-run it."""
    db = get_supabase_client()
    try:
        resp = await asyncio.to_thread(
            lambda: db.client.table("integration_actions")
            .update({"status": "running", "attempts": attempts + 1, "updated_at": _now_iso()})
            .eq("id", action_id)
            .in_("status", ["queued", "confirmed"])
            .execute()
        )
        return bool(resp.data)
    except Exception as e:
        logger.error(f"[dispatcher] claim failed for {action_id}: {e}")
        return False


async def _finalize(action_id: str, status: str, result: dict | None = None, error: str | None = None) -> None:
    db = get_supabase_client()
    update = {"status": status, "updated_at": _now_iso()}
    if result is not None:
        update["result"] = result
    if error is not None:
        update["error_message"] = error
    try:
        await asyncio.to_thread(
            lambda: db.client.table("integration_actions").update(update).eq("id", action_id).execute()
        )
    except Exception as e:
        logger.error(f"[dispatcher] finalize failed for {action_id}: {e}")


def _derive_match_key(action_type: str, payload: dict) -> str | None:
    """Derive a stable match key for autonomy graduation logging."""
    if action_type in ("reply_to_email", "send_email"):
        addr = (payload.get("recipient") or payload.get("to") or "").strip().lower()
        return addr.split("@", 1)[1] if "@" in addr else None
    if action_type == "send_slack_message":
        return (payload.get("channel") or "").strip().lower() or None
    return None


async def _log_approval_outcome(user_id: str, action_type: str, payload: dict, outcome: str) -> None:
    """Insert a row into action_approval_log to track a confirmed or discarded action."""
    db = get_supabase_client()
    match_key = _derive_match_key(action_type, payload)
    try:
        await asyncio.to_thread(
            lambda: db.client.table("action_approval_log").insert({
                "user_id": user_id,
                "action_type": action_type,
                "match_key": match_key,
                "outcome": outcome,
            }).execute()
        )
    except Exception as e:
        logger.warning(f"[dispatcher] approval log insert failed ({action_type}, {outcome}): {e}")


async def dispatch_action(action: dict) -> None:
    """Execute a single action row (already fetched). Honors the confirmation gate."""
    action_id = action["id"]
    provider = action.get("provider", "")
    action_type = action.get("action_type", "")
    attempts = action.get("attempts", 0) or 0
    user_id = action.get("user_id", "")
    payload = action.get("payload") or {}

    # Safety chokepoint: never execute an unconfirmed risky action.
    if action.get("requires_confirmation") and not action.get("confirmed_at"):
        logger.info(f"[dispatcher] {action_id} awaiting confirmation; skipping")
        return

    if not await _claim_action(action_id, attempts):
        return  # someone else claimed it, or it left the actionable set

    logger.info(f"[dispatcher] executing {action_type} ({provider}) id={action_id}")

    try:
        if provider in _N8N_PROVIDERS:
            ok, result, error = await _dispatch_to_n8n(action)
            if ok:
                # n8n writes the terminal status itself; leave it 'running' only
                # if it didn't. Mark done defensively if no callback expected.
                logger.info(f"[dispatcher] {action_id} handed to n8n")
                await _log_approval_outcome(user_id, action_type, payload, "confirmed")
            else:
                await _maybe_retry_or_fail(action_id, attempts, error or "n8n dispatch failed")
            return

        handler = NATIVE_HANDLERS.get(action_type)
        if not handler:
            await _finalize(action_id, "failed", error=f"No native handler for '{action_type}'")
            return

        ctx = await _build_tool_context(user_id)
        result_str = await handler(ctx, payload)
        await _finalize(action_id, "done", result={"message": result_str})
        await _log_approval_outcome(user_id, action_type, payload, "confirmed")
        logger.info(f"[dispatcher] {action_id} done: {result_str[:120]}")
    except Exception as e:
        logger.error(f"[dispatcher] error executing {action_id}: {e}", exc_info=True)
        await _maybe_retry_or_fail(action_id, attempts, str(e))


async def _maybe_retry_or_fail(action_id: str, attempts: int, error: str) -> None:
    if attempts + 1 >= _MAX_ATTEMPTS:
        await _finalize(action_id, "failed", error=error)
    else:
        # Drop back to 'queued' so the next poll retries it.
        db = get_supabase_client()
        await asyncio.to_thread(
            lambda: db.client.table("integration_actions")
            .update({"status": "queued", "error_message": error, "updated_at": _now_iso()})
            .eq("id", action_id)
            .execute()
        )


async def dispatch_action_by_id(action_id: str) -> None:
    db = get_supabase_client()
    try:
        resp = await asyncio.to_thread(
            lambda: db.client.table("integration_actions").select("*").eq("id", action_id).maybe_single().execute()
        )
        if resp and resp.data:
            await dispatch_action(resp.data)
    except Exception as e:
        logger.error(f"[dispatcher] dispatch_action_by_id {action_id} failed: {e}")


async def run_due_actions(limit: int = 20) -> int:
    """Poll for actionable rows and dispatch them. Returns the count dispatched.

    Actionable = status in (queued, confirmed), not waiting on confirmation, and
    either unscheduled or scheduled_for <= now.
    """
    db = get_supabase_client()
    now_iso = _now_iso()
    try:
        resp = await asyncio.to_thread(
            lambda: db.client.table("integration_actions")
            .select("*")
            .in_("status", ["queued", "confirmed"])
            .order("created_at", desc=False)
            .limit(limit)
            .execute()
        )
        rows = resp.data or []
    except Exception as e:
        logger.error(f"[dispatcher] poll query failed: {e}")
        return 0

    dispatched = 0
    for action in rows:
        if action.get("requires_confirmation") and not action.get("confirmed_at"):
            continue
        scheduled_for = action.get("scheduled_for")
        if scheduled_for and scheduled_for > now_iso:
            continue
        await dispatch_action(action)
        dispatched += 1

    if dispatched:
        logger.info(f"[dispatcher] poll dispatched {dispatched} action(s)")
    return dispatched


# ---------------------------------------------------------------------------
# Approval notifications (descriptive push, deep-linked to the review screen)
# ---------------------------------------------------------------------------

_ACTION_TITLES = {
    "reply_to_email": "Approve email reply",
    "send_email": "Approve email",
    "create_notion_page": "Approve Notion page",
    "update_notion_page": "Approve Notion update",
    "send_slack_message": "Approve Slack message",
}


async def notify_pending_approvals(limit: int = 30) -> int:
    """Send a descriptive push for any pending_confirmation action that hasn't
    been announced yet, deep-linking to the in-app review screen."""
    db = get_supabase_client()
    try:
        resp = await asyncio.to_thread(
            lambda: db.client.table("integration_actions")
            .select("id, user_id, action_type, summary")
            .eq("status", "pending_confirmation")
            .is_("notified_at", "null")
            .limit(limit)
            .execute()
        )
        rows = resp.data or []
    except Exception as e:
        logger.error(f"[dispatcher] approval-notify query failed: {e}")
        return 0

    sent = 0
    for action in rows:
        action_id = action["id"]
        user_id = action["user_id"]
        title = _ACTION_TITLES.get(action.get("action_type", ""), "Approve action")
        body = action.get("summary") or "Praxa drafted something that needs your approval."
        try:
            token = await get_user_push_token(user_id)
            if token:
                ticket_id = await send_push_notification(
                    push_token=token,
                    title=title,
                    body=body,
                    data={"notificationType": "action_review", "actionId": action_id},
                )
                if ticket_id:
                    schedule_receipt_check(ticket_id, user_id)
            # Mark notified regardless of token presence so we don't re-poll forever.
            await asyncio.to_thread(
                lambda aid=action_id: db.client.table("integration_actions")
                .update({"notified_at": _now_iso()})
                .eq("id", aid)
                .execute()
            )
            sent += 1
        except Exception as e:
            logger.error(f"[dispatcher] approval-notify failed for {action_id}: {e}")

    if sent:
        logger.info(f"[dispatcher] sent {sent} approval notification(s)")
    return sent
