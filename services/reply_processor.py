
import json
import logging
import os
import re
from datetime import date, datetime
from typing import Optional

from openai import AsyncOpenAI

from .supabase_client import get_supabase_client

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are Praxa, a productivity assistant. A user replied to an {channel} notification.

The notification they received was about: {notification_context}

Their current task state:
{context}

Understand what they want naturally — don't require exact phrasing.
Reply with valid JSON only, no markdown.

Resolution rules:
- "done", "finished", "completed": complete_task. If exactly one task was in the notification, \
use that task. If multiple, pick the best match from the message.
- Partial progress in any domain, like "made progress", "started it", "did some of it", \
"hit part of the target", "not finished yet": partial_progress. Do NOT complete the task; capture the progress note and ask whether to finish later, resize, or break it down.
- "snooze", "push", "move", "later", "reschedule [task]": snooze_task. Parse relative dates \
(tomorrow={tomorrow}, Friday={next_friday}, next week={next_monday}) into YYYY-MM-DD.
- "cancel [call]", "skip [call]", "not today": reschedule_call.
- "add", "create", "remind me to": add_task.
- "status", "what do I have", "what's on my list": get_status.
- "ok", "thanks", "got it": acknowledge.
- "stop", "pause", "no more": stop_notifications.

CRITICAL rules for reply_message:
- NEVER tell the user to open the app. Everything must be completable by replying.
- If you need clarification (multiple matching tasks), ask specifically: \
"Which task? Reply with the name."
- Keep replies under 160 chars for SMS. Be warm and direct.
- On success, confirm exactly what happened: "Done -- 'Client deck' marked complete."
- On partial_progress, acknowledge progress and ask one short adjustment question, e.g. "Noted progress on that. Finish later, resize it, or split it up?"
- On get_status, list the tasks inline — don't just say "you have tasks".

{{
  "intent": "complete_task|partial_progress|snooze_task|add_task|reschedule_call|get_status|acknowledge|stop_notifications|unknown",
  "confidence": <0.0-1.0>,
  "params": {{"task_id": "<optional>", "best_match_title": "<optional>", "progress_note": "<optional summary of what happened>"}},
  "reply_message": "<self-contained reply>"
}}
"""


def _get_openai() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


async def _fetch_user_context(user_id: str) -> dict:
    """Load the user's tasks, next call, preferences, and last notification from Supabase."""
    db = get_supabase_client()
    today = date.today().isoformat()

    try:
        tasks_resp = db.client.table("loops").select(
            "id, title, due_date, status, is_this_week, bucket_id"
        ).eq("user_id", user_id).eq("status", "open").eq("archived", False).limit(25).execute()
        open_tasks: list[dict] = tasks_resp.data or []

        call_resp = db.client.table("scheduled_calls").select(
            "id, scheduled_for, status"
        ).eq("user_id", user_id).eq("status", "pending").order(
            "scheduled_for", desc=False
        ).limit(1).execute()
        upcoming_call: Optional[dict] = call_resp.data[0] if call_resp.data else None

        settings_resp = db.client.table("user_settings").select(
            "timezone, sprint_cadence, notification_preferences"
        ).eq("user_id", user_id).maybeSingle().execute()
        settings: dict = settings_resp.data or {}

        log_resp = db.client.table("notification_log").select(
            "dedupe_key, event_type, channel, sent_at"
        ).eq("user_id", user_id).order("sent_at", desc=True).limit(1).execute()
        last_notification: Optional[dict] = log_resp.data[0] if log_resp.data else None

    except Exception as exc:
        logger.error(f"[ReplyProcessor] Context fetch failed for {user_id}: {exc}")
        return {
            "context_text": f"Today: {today}. Context unavailable.",
            "open_tasks": [],
            "upcoming_call": None,
            "settings": {},
            "last_notification": None,
        }

    lines = [f"Today: {today}"]

    due_today = [t for t in open_tasks if t.get("due_date") == today]
    overdue = [t for t in open_tasks if t.get("due_date") and t["due_date"] < today]
    sprint = [t for t in open_tasks if t.get("is_this_week")]

    def _fmt(tasks: list[dict], limit: int = 6) -> str:
        return ", ".join(
            f'"{t["title"]}" (id:{t["id"][:8]})' for t in tasks[:limit]
        )

    if due_today:
        lines.append(f"Due today ({len(due_today)}): {_fmt(due_today)}")
    if overdue:
        lines.append(f"Overdue ({len(overdue)}): {_fmt(overdue)}")
    if sprint:
        lines.append(f"This sprint ({len(sprint)}): {_fmt(sprint)}")
    if not open_tasks:
        lines.append("No open tasks.")

    if upcoming_call:
        lines.append(
            f"Next check-in call: {upcoming_call.get('scheduled_for', '?')} "
            f"(id:{upcoming_call['id'][:8]})"
        )

    return {
        "context_text": "\n".join(lines),
        "open_tasks": open_tasks,
        "upcoming_call": upcoming_call,
        "settings": settings,
        "last_notification": last_notification,
    }


def _notification_age_hours(last_notification: Optional[dict]) -> Optional[float]:
    """Return how many hours ago the last notification was sent, or None if unknown."""
    if not last_notification:
        return None
    try:
        sent_at = datetime.fromisoformat(
            last_notification["sent_at"].replace("Z", "+00:00")
        )
        delta = datetime.now(sent_at.tzinfo) - sent_at
        return delta.total_seconds() / 3600
    except Exception:
        return None


async def _validate_context(
    user_id: str,
    intent: str,
    params: dict,
    ctx: dict,
    channel: str,
) -> tuple[bool, str]:
    """
    Check whether the action is still valid given the current app state.

    Returns (can_proceed, override_reply).
    - can_proceed=True  → run the action normally.
    - can_proceed=False → skip execution and return override_reply to the user.

    Checks performed:
      1. Task-targeting intents: verify the task is still open.
         If already completed  → celebrate and skip.
         If missing/archived   → inform the user.
      2. Call intents: verify a pending call still exists.
      3. Stale-notification warning: if the notification is older than the
         channel threshold and the entity is gone, add context to the reply.
    """
    db = get_supabase_client()
    open_tasks: list[dict] = ctx.get("open_tasks", [])
    upcoming_call: Optional[dict] = ctx.get("upcoming_call")
    age_hours = _notification_age_hours(ctx.get("last_notification"))

    stale_threshold = 6.0 if channel == "sms" else 48.0
    is_stale = age_hours is not None and age_hours > stale_threshold

    if intent in ("complete_task", "partial_progress", "snooze_task"):
        hint = params.get("task_id") or params.get("best_match_title", "")
        if hint and not _find_task(open_tasks, hint):
            try:
                done_resp = db.client.table("loops").select(
                    "title, status"
                ).eq("user_id", user_id).in_(
                    "status", ["completed"]
                ).limit(30).execute()
                completed_tasks: list[dict] = done_resp.data or []
                hl = hint.lower()
                match = next(
                    (t for t in completed_tasks if hl in t["title"].lower()), None
                )
            except Exception:
                match = None

            if match:
                return False, f'That task ("{match["title"][:60]}") was already marked done — nice work!'
            if is_stale:
                return (
                    False,
                    "It looks like things may have changed since that notification — "
                    "that task no longer appears in your open list. "
                    "Reply 'status' to see your current tasks.",
                )
            return False, "I couldn't find that task in your open list. Reply 'status' to see your tasks."

    if intent == "reschedule_call" and not upcoming_call:
        return (
            False,
            "I couldn't find a pending call to reschedule — it may have already "
            "been completed or cancelled. Reply with a new time if you want to schedule one.",
        )

    return True, ""


def _build_notification_context(last_notification: Optional[dict], open_tasks: list[dict]) -> str:
    """
    Reconstruct a human-readable description of what the last notification was about,
    so the AI can resolve bare references like "done" or "that one".
    """
    if not last_notification:
        return "unknown (no recent notification found)"

    event = last_notification.get("event_type", "")
    today = date.today().isoformat()

    if event == "task_due_today":
        tasks = [t for t in open_tasks if t.get("due_date") == today]
        if tasks:
            names = ", ".join(f'"{t["title"]}"' for t in tasks[:3])
            return f"tasks due today: {names}"
        return "tasks due today"

    if event == "task_overdue":
        tasks = [t for t in open_tasks if t.get("due_date") and t["due_date"] < today]
        if tasks:
            names = ", ".join(f'"{t["title"]}"' for t in tasks[:3])
            return f"overdue tasks: {names}"
        return "overdue tasks"

    if event == "sprint_deadline":
        tasks = [t for t in open_tasks if t.get("is_this_week")]
        if tasks:
            names = ", ".join(f'"{t["title"]}"' for t in tasks[:3])
            return f"sprint tasks still open today: {names}"
        return "sprint ending today"

    if event == "call_reminder":
        return "an upcoming check-in call in 5 minutes"
    if event == "call_missed":
        return "a missed check-in call"
    if event == "call_completed":
        return "a completed check-in call"
    if event == "goal_nudge":
        return "sprint tasks tied to a goal"
    if event == "task_review_idle":
        tasks = open_tasks[:3]
        if tasks:
            names = ", ".join(f'"{t["title"]}"' for t in tasks)
            return f"open tasks you haven't reviewed recently: {names}"
        return "open tasks you haven't reviewed recently"

    return event.replace("_", " ")


def _date_hints() -> dict:
    """Compute concrete dates for relative-date placeholders in the AI prompt."""
    today = date.today()
    weekday = today.weekday()  # 0=Mon … 6=Sun
    days_to_friday = (4 - weekday) % 7 or 7
    days_to_monday = (7 - weekday) % 7 or 7
    return {
        "tomorrow": (today.replace() if False else
                     date.fromordinal(today.toordinal() + 1)).isoformat(),
        "next_friday": date.fromordinal(today.toordinal() + days_to_friday).isoformat(),
        "next_monday": date.fromordinal(today.toordinal() + days_to_monday).isoformat(),
    }


async def _parse_intent(message: str, user_context: dict, channel: str) -> dict:
    """Call GPT-4o-mini to classify the user's reply."""
    openai = _get_openai()
    dates = _date_hints()
    notification_ctx = _build_notification_context(
        user_context.get("last_notification"),
        user_context.get("open_tasks", []),
    )
    system = _SYSTEM_PROMPT.format(
        channel=channel,
        notification_context=notification_ctx,
        context=user_context["context_text"],
        tomorrow=dates["tomorrow"],
        next_friday=dates["next_friday"],
        next_monday=dates["next_monday"],
    )
    try:
        resp = await openai.chat.completions.create(
            model="gpt-5.4-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": message},
            ],
            temperature=0.1,
            max_tokens=350,
        )
        raw = (resp.choices[0].message.content or "{}").strip()
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1].lstrip("json").strip() if len(parts) > 1 else "{}"
        return json.loads(raw)
    except Exception as exc:
        logger.error(f"[ReplyProcessor] OpenAI parse failed: {exc}")
        return {
            "intent": "unknown",
            "confidence": 0.0,
            "params": {},
            "reply_message": "Got it. Reply 'status' to see your tasks or send a task update.",
        }


def _find_task(tasks: list[dict], hint: str) -> Optional[dict]:
    """Match a task by 8-char ID prefix or case-insensitive title substring."""
    if not hint:
        return None
    for t in tasks:
        if t["id"].startswith(hint) or t["id"][:8] == hint:
            return t
    hint_lower = hint.lower()
    for t in tasks:
        if hint_lower in t["title"].lower():
            return t
    return None


def _append_task_note(user_id: str, task_id: str, note: str, source: str) -> None:
    """Append a Praxa-authored note to a task without changing its completion state."""
    db = get_supabase_client()
    existing = db.client.table("loops").select("notes").eq(
        "id", task_id
    ).eq("user_id", user_id).single().execute()

    existing_notes = existing.data.get("notes", "") if existing.data else ""
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
    note_line = f"[{timestamp} - Praxa {source}] {note}"
    new_notes = f"{existing_notes}\n\n{note_line}" if existing_notes else note_line

    db.client.table("loops").update({
        "notes": new_notes,
        "updated_at": datetime.utcnow().isoformat(),
    }).eq("id", task_id).eq("user_id", user_id).execute()


async def _execute(
    user_id: str, parsed: dict, ctx: dict, channel: str
) -> tuple[str, dict, Optional[str]]:
    """
    Validate context then run the action.

    Returns (action_taken, action_result, override_reply).
    override_reply is set when context validation blocks execution so the
    caller can send a more informative message to the user instead of the
    AI-generated one.
    """
    db = get_supabase_client()
    intent: str = parsed.get("intent", "unknown")
    params: dict = parsed.get("params", {})
    open_tasks: list[dict] = ctx.get("open_tasks", [])

    can_proceed, override_reply = await _validate_context(
        user_id, intent, params, ctx, channel
    )
    if not can_proceed:
        return "context_invalid", {}, override_reply

    try:
        if intent == "complete_task":
            hint = params.get("task_id") or params.get("best_match_title", "")
            task = _find_task(open_tasks, hint)
            if not task:
                return "task_not_found", {"hint": hint}, None
            db.client.table("loops").update(
                {"status": "completed", "updated_at": datetime.utcnow().isoformat()}
            ).eq("id", task["id"]).eq("user_id", user_id).execute()
            return "task_completed", {"task_id": task["id"], "title": task["title"]}, None

        if intent == "partial_progress":
            hint = params.get("task_id") or params.get("best_match_title", "")
            task = _find_task(open_tasks, hint)
            if not task:
                return "task_not_found", {"hint": hint}, None

            progress_note = params.get("progress_note", "").strip()
            if not progress_note:
                progress_note = "User reported partial progress via reply; task remains open."
            _append_task_note(user_id, task["id"], progress_note, channel.upper())
            return "partial_progress_noted", {
                "task_id": task["id"],
                "title": task["title"],
                "progress_note": progress_note,
            }, None

        if intent == "snooze_task":
            hint = params.get("task_id") or params.get("best_match_title", "")
            new_due = params.get("new_due_date", "")
            task = _find_task(open_tasks, hint)
            if not task or not new_due:
                return "snooze_failed", {"hint": hint, "new_due_date": new_due}, None
            db.client.table("loops").update(
                {"due_date": new_due, "updated_at": datetime.utcnow().isoformat()}
            ).eq("id", task["id"]).eq("user_id", user_id).execute()
            return "task_snoozed", {
                "task_id": task["id"],
                "title": task["title"],
                "new_due_date": new_due,
            }, None

        if intent == "add_task":
            title = params.get("title", "").strip()
            if not title:
                return "add_failed", {}, None
            new_row: dict = {
                "user_id": user_id,
                "title": title,
                "status": "open",
                "archived": False,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
            }
            if params.get("due_date"):
                new_row["due_date"] = params["due_date"]
            res = db.client.table("loops").insert(new_row).execute()
            new_id = res.data[0]["id"] if res.data else None
            return "task_added", {"task_id": new_id, "title": title}, None

        if intent == "reschedule_call":
            call = ctx.get("upcoming_call")
            if not call:
                return "no_call_found", {}, None
            db.client.table("scheduled_calls").update(
                {"status": "cancelled", "updated_at": datetime.utcnow().isoformat()}
            ).eq("id", call["id"]).execute()
            return "call_cancelled", {"call_id": call["id"]}, None

        if intent == "get_status":
            tasks = open_tasks[:8]
            summary = "; ".join(t["title"] for t in tasks) if tasks else "No open tasks"
            return "get_status", {"summary": summary}, None

        if intent in ("acknowledge", "stop_notifications", "unknown"):
            return intent, {}, None

    except Exception as exc:
        logger.error(f"[ReplyProcessor] Action {intent} failed for {user_id}: {exc}")
        return "execution_error", {"error": str(exc)}, None

    return "unknown", {}, None


async def _log_reply(
    user_id: str,
    channel: str,
    raw_message: str,
    intent: str,
    action_taken: str,
    action_result: dict,
    reply_sent: str,
    triggered_by_dedupe_key: Optional[str],
    error: Optional[str],
) -> None:
    db = get_supabase_client()
    try:
        db.client.table("notification_replies").insert({
            "user_id": user_id,
            "channel": channel,
            "raw_message": raw_message,
            "triggered_by_dedupe_key": triggered_by_dedupe_key,
            "intent": intent,
            "action_taken": action_taken,
            "action_result": action_result,
            "reply_sent": reply_sent,
            "processed_at": datetime.utcnow().isoformat(),
            "error": error,
        }).execute()
    except Exception as exc:
        logger.error(f"[ReplyProcessor] Failed to log reply: {exc}")


async def process_reply(
    user_id: str,
    channel: str,
    raw_message: str,
    triggered_by_dedupe_key: Optional[str] = None,
) -> str:
    """
    Full pipeline: context → intent → action → log → reply text.

    Returns the short string to send back to the user (≤ 160 chars for SMS).
    """
    ctx = await _fetch_user_context(user_id)
    parsed = await _parse_intent(raw_message, ctx, channel)

    intent: str = parsed.get("intent", "unknown")
    reply: str = parsed.get(
        "reply_message", "Got it. Reply 'status' to see your tasks or send a task update."
    )

    action_taken, action_result, override_reply = await _execute(
        user_id, parsed, ctx, channel
    )

    if override_reply:
        reply = override_reply
    elif action_taken == "partial_progress_noted":
        title = action_result.get("title", "that task")
        reply = f"Noted progress on '{title}'. Finish later, lower target, or split it up?"
        if len(reply) > 160:
            reply = reply[:157] + "..."
    elif action_taken == "get_status":
        summary: str = action_result.get("summary", "No open tasks")
        reply = f"Your open tasks: {summary}"
        if len(reply) > 160:
            reply = reply[:157] + "..."

    await _log_reply(
        user_id=user_id,
        channel=channel,
        raw_message=raw_message,
        intent=intent,
        action_taken=action_taken,
        action_result=action_result,
        reply_sent=reply,
        triggered_by_dedupe_key=triggered_by_dedupe_key,
        error=None,
    )

    logger.info(
        f"[ReplyProcessor] user={user_id} channel={channel} "
        f"intent={intent} action={action_taken}"
    )
    return reply


async def lookup_user_by_phone(from_number: str) -> Optional[str]:
    """
    Find a user_id from an inbound E.164 phone number.
    Matches against user_settings rows that have sms_notifications_opt_in=true
    and phone_verified=true.
    """
    db = get_supabase_client()
    try:
        rows = db.client.table("user_settings").select(
            "user_id, phone_number, phone_country_code"
        ).eq("sms_notifications_opt_in", True).eq("phone_verified", True).execute()

        from_digits = re.sub(r"\D", "", from_number)
        for row in rows.data or []:
            stored = re.sub(r"\D", "", row.get("phone_number") or "")
            if stored and stored in from_digits:
                return row["user_id"]
    except Exception as exc:
        logger.error(f"[ReplyProcessor] Phone lookup failed: {exc}")
    return None
