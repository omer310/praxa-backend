"""Daily briefing generator.

Builds a once-daily synthesized brief per user from the signals Praxa already
tracks (attention-worthy email, today's calendar, tasks, open matters, pending
approvals), persists it to `daily_briefs`, and knocks via push (+ optional SMS)
that deep-links into the in-app brief.

Design notes:
  - The structured `content` (sections + deep-link routes) is assembled
    deterministically so the mobile sheet renders predictably; the LLM is used
    ONLY to write the natural-language `summary`. If OpenAI is unavailable we
    fall back to a templated summary.
  - Status is `ready` when there's something worth surfacing, else `empty`
    (no push). The mobile surface only shows `ready` + unviewed briefs.
  - Scheduling mirrors the per-user local-hour pattern from the task nudges:
    the job runs hourly and fires for a user when it's their BRIEFING_HOUR and
    no brief exists yet for their local date.
"""
import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from openai import AsyncOpenAI

from .proactive import flag_enabled, is_allowed
from .push_service import send_push_notification, get_user_push_token, schedule_receipt_check
from .supabase_client import get_supabase_client

logger = logging.getLogger(__name__)

_BRIEFING_MODEL = os.getenv("BRIEFING_MODEL", "gpt-4o-mini")
_EMPTY_HINTS = ("didn't find", "don't see", "no events", "couldn't", "i can only")


def _enabled() -> bool:
    return flag_enabled("PROACTIVE_BRIEFING_ENABLED", default=True)


def _briefing_hour() -> int:
    try:
        return max(0, min(23, int(os.getenv("BRIEFING_HOUR", "8"))))
    except (TypeError, ValueError):
        return 8


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _openai() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ---------------------------------------------------------------------------
# Tool context (calendar reads need a praxa_core ToolContext)
# ---------------------------------------------------------------------------

async def _build_ctx(user_id: str, timezone_str: str):
    from praxa_core import ToolContext

    db = get_supabase_client()
    return ToolContext(
        user_id=user_id,
        supabase=db.client,
        timezone=timezone_str or "UTC",
        surface="briefing",
        nylas_api_key=os.getenv("NYLAS_API_KEY"),
    )


# ---------------------------------------------------------------------------
# Signal gathering
# ---------------------------------------------------------------------------

async def _attention_emails(user_id: str, limit: int = 8) -> list[dict]:
    db = get_supabase_client()
    try:
        resp = await asyncio.to_thread(
            lambda: db.client.table("email_insights")
            .select("email_id, from_name, from_email, subject, snippet, priority_score")
            .eq("user_id", user_id)
            .eq("insight_type", "needs_attention")
            .eq("is_addressed", False)
            .order("priority_score", desc=True)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return resp.data or []
    except Exception as e:
        logger.warning(f"[briefing] attention emails failed for {user_id}: {e}")
        return []


async def _pending_approvals(user_id: str, limit: int = 8) -> list[dict]:
    db = get_supabase_client()
    try:
        resp = await asyncio.to_thread(
            lambda: db.client.table("integration_actions")
            .select("id, action_type, summary")
            .eq("user_id", user_id)
            .eq("status", "pending_confirmation")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return resp.data or []
    except Exception as e:
        logger.warning(f"[briefing] pending approvals failed for {user_id}: {e}")
        return []


async def _open_matters(user_id: str, limit: int = 5) -> list[dict]:
    db = get_supabase_client()
    try:
        resp = await asyncio.to_thread(
            lambda: db.client.table("user_matters")
            .select("id, title, status, last_activity_at")
            .eq("user_id", user_id)
            .order("last_activity_at", desc=True)
            .limit(limit + 5)
            .execute()
        )
        rows = resp.data or []
        return [r for r in rows if (r.get("status") or "open") not in ("closed", "done", "archived")][:limit]
    except Exception as e:
        logger.warning(f"[briefing] matters failed for {user_id}: {e}")
        return []


async def _calendar_text(user_id: str, timezone_str: str) -> str:
    try:
        from praxa_core.tools import calendar as _calendar
        ctx = await _build_ctx(user_id, timezone_str)
        text = await _calendar.get_todays_events_impl(ctx)
        return (text or "").strip()
    except Exception as e:
        logger.warning(f"[briefing] calendar failed for {user_id}: {e}")
        return ""


def _looks_empty(text: str) -> bool:
    low = text.lower()
    return (not text) or any(h in low for h in _EMPTY_HINTS)


# ---------------------------------------------------------------------------
# Brief assembly
# ---------------------------------------------------------------------------

def _build_sections(
    emails: list[dict],
    approvals: list[dict],
    calendar_text: str,
    due_today: list[dict],
    overdue: list[dict],
    sprint: list[dict],
    matters: list[dict],
) -> list[dict]:
    sections: list[dict] = []

    if emails:
        sections.append({
            "key": "needs_you",
            "title": "Needs you",
            "icon": "envelope.fill",
            "items": [
                {
                    "label": (e.get("subject") or "(no subject)")[:120],
                    "sublabel": f"from {e.get('from_name') or e.get('from_email') or 'someone'}",
                    "route": "/email-mode",
                    "ref": e.get("email_id"),
                }
                for e in emails
            ],
        })

    if approvals:
        sections.append({
            "key": "approvals",
            "title": "Awaiting your approval",
            "icon": "sparkles",
            "items": [
                {
                    "label": a.get("summary") or (a.get("action_type") or "Pending action").replace("_", " "),
                    "route": "/action-review",
                    "ref": a.get("id"),
                    "query": {"actionId": a.get("id")},
                }
                for a in approvals
            ],
        })

    if calendar_text and not _looks_empty(calendar_text):
        lines = [ln.strip("-• ") for ln in calendar_text.splitlines() if ln.strip()]
        sections.append({
            "key": "calendar",
            "title": "Today",
            "icon": "calendar",
            "items": [{"label": ln[:140], "route": "/calendar-mode"} for ln in lines[:6]],
        })

    task_items: list[dict] = []
    for t in overdue[:5]:
        task_items.append({"label": t.get("title") or "Task", "sublabel": "Overdue", "route": "/(tabs)/initiatives"})
    for t in due_today[:5]:
        task_items.append({"label": t.get("title") or "Task", "sublabel": "Due today", "route": "/(tabs)/initiatives"})
    if task_items:
        sections.append({"key": "tasks", "title": "Tasks", "icon": "checkmark.circle", "items": task_items})
    elif sprint:
        sections.append({
            "key": "tasks",
            "title": "This week's sprint",
            "icon": "checkmark.circle",
            "items": [{"label": f"{len(sprint)} open task{'s' if len(sprint) != 1 else ''} this sprint", "route": "/(tabs)/initiatives"}],
        })

    if matters:
        sections.append({
            "key": "matters",
            "title": "Matters moving",
            "icon": "person.3.fill",
            "items": [{"label": (m.get("title") or "Matter")[:120], "route": "/crm-mode", "ref": m.get("id")} for m in matters],
        })

    return sections


def _fallback_summary(highlights: dict) -> str:
    parts = []
    if highlights.get("emails"):
        parts.append(f"{highlights['emails']} email{'s' if highlights['emails'] != 1 else ''} need you")
    if highlights.get("approvals"):
        parts.append(f"{highlights['approvals']} awaiting approval")
    if highlights.get("meetings"):
        parts.append(f"{highlights['meetings']} on your calendar")
    if highlights.get("tasks"):
        parts.append(f"{highlights['tasks']} task{'s' if highlights['tasks'] != 1 else ''} to handle")
    if highlights.get("matters"):
        parts.append(f"{highlights['matters']} matter{'s' if highlights['matters'] != 1 else ''} moving")
    if not parts:
        return "You're all clear — nothing pressing this morning."
    return "Here's your day: " + ", ".join(parts) + "."


async def _compose_summary(sections: list[dict], highlights: dict) -> str:
    if not sections:
        return "You're all clear — nothing pressing this morning."
    if not os.getenv("OPENAI_API_KEY"):
        return _fallback_summary(highlights)

    digest_lines = []
    for s in sections:
        items = "; ".join(i.get("label", "") for i in s.get("items", [])[:5])
        digest_lines.append(f"{s['title']}: {items}")
    prompt = (
        "Write a brief, warm morning summary for a busy professional based on the signals below. "
        "2-3 short sentences, plain text, no greeting, no markdown, no emojis. Be specific and concrete. "
        "Lead with what most needs their attention.\n\n" + "\n".join(digest_lines)
    )
    try:
        resp = await _openai().chat.completions.create(
            model=_BRIEFING_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=160,
        )
        text = (resp.choices[0].message.content or "").strip()
        return text or _fallback_summary(highlights)
    except Exception as e:
        logger.warning(f"[briefing] summary LLM failed: {e}")
        return _fallback_summary(highlights)


async def _user_timezone(user_id: str) -> str:
    db = get_supabase_client()
    try:
        resp = await asyncio.to_thread(
            lambda: db.client.table("user_settings").select("timezone").eq("user_id", user_id).maybe_single().execute()
        )
        if resp and resp.data and resp.data.get("timezone"):
            return resp.data["timezone"]
    except Exception:
        pass
    return "UTC"


async def _maybe_send_sms(user_id: str, summary: str) -> None:
    db = get_supabase_client()
    try:
        resp = await asyncio.to_thread(
            lambda: db.client.table("user_settings")
            .select("phone_number, phone_verified, sms_notifications_opt_in")
            .eq("user_id", user_id).maybe_single().execute()
        )
        row = (resp.data if resp else None) or {}
        if row.get("sms_notifications_opt_in") and row.get("phone_verified") and row.get("phone_number"):
            from .twilio_sms import send_sms
            await send_sms(row["phone_number"], f"Praxa brief: {summary[:280]} Reply for detail or open the app.")
    except Exception as e:
        logger.warning(f"[briefing] sms send failed for {user_id}: {e}")


async def generate_brief(user_id: str, *, timezone_str: str | None = None, send_notifications: bool = True) -> dict:
    """Generate (or refresh) today's brief for a user. Returns the persisted row dict."""
    db = get_supabase_client()
    tz = timezone_str or await _user_timezone(user_id)
    try:
        local_date = datetime.now(ZoneInfo(tz)).date()
    except (ZoneInfoNotFoundError, Exception):
        local_date = _now().date()

    emails = await _attention_emails(user_id)
    approvals = await _pending_approvals(user_id)
    calendar_text = await _calendar_text(user_id, tz)
    try:
        due_today = await db.get_tasks_due_today(user_id, tz)
    except Exception:
        due_today = []
    try:
        overdue = await db.get_overdue_tasks(user_id)
    except Exception:
        overdue = []
    try:
        sprint = await db.get_this_week_tasks(user_id)
    except Exception:
        sprint = []
    matters = await _open_matters(user_id)

    sections = _build_sections(emails, approvals, calendar_text, due_today, overdue, sprint, matters)
    meetings_count = next((len(s["items"]) for s in sections if s["key"] == "calendar"), 0)
    highlights = {
        "emails": len(emails),
        "approvals": len(approvals),
        "meetings": meetings_count,
        "tasks": len(overdue) + len(due_today),
        "matters": len(matters),
    }
    summary = await _compose_summary(sections, highlights)
    status = "ready" if sections else "empty"
    now_iso = _now().isoformat()
    expires_at = (_now() + timedelta(hours=40)).isoformat()

    row = {
        "user_id": user_id,
        "brief_date": local_date.isoformat(),
        "status": status,
        "summary": summary,
        "content": {"sections": sections},
        "highlights": highlights,
        "ready_at": now_iso if status == "ready" else None,
        "expires_at": expires_at,
    }
    try:
        await asyncio.to_thread(
            lambda: db.client.table("daily_briefs").upsert(row, on_conflict="user_id,brief_date").execute()
        )
    except Exception as e:
        logger.error(f"[briefing] upsert failed for {user_id}: {e}")
        return row

    if send_notifications and status == "ready":
        await _notify(user_id, summary, highlights)
        await _maybe_send_sms(user_id, summary)

    logger.info(f"[briefing] generated {status} brief for {user_id} ({local_date})")
    return row


async def _notify(user_id: str, summary: str, highlights: dict) -> None:
    try:
        token = await get_user_push_token(user_id)
        if not token:
            return
        body = summary if len(summary) <= 178 else summary[:175] + "..."
        ticket_id = await send_push_notification(
            push_token=token,
            title="Your morning brief",
            body=body,
            data={"notificationType": "daily_brief"},
        )
        if ticket_id:
            schedule_receipt_check(ticket_id, user_id)
    except Exception as e:
        logger.warning(f"[briefing] push failed for {user_id}: {e}")


# ---------------------------------------------------------------------------
# Scheduler entry point
# ---------------------------------------------------------------------------

async def _brief_exists_today(user_id: str, local_date_iso: str) -> bool:
    db = get_supabase_client()
    try:
        resp = await asyncio.to_thread(
            lambda: db.client.table("daily_briefs")
            .select("id").eq("user_id", user_id).eq("brief_date", local_date_iso).limit(1).execute()
        )
        return bool(resp.data)
    except Exception:
        return False


async def run_daily_briefing() -> int:
    """Hourly job: generate today's brief for each user whose local hour matches
    BRIEFING_HOUR and who doesn't have one yet. Returns count generated."""
    if not _enabled():
        return 0

    db = get_supabase_client()
    try:
        users = await db.get_all_users_with_push_tokens()
    except Exception as e:
        logger.error(f"[briefing] user fetch failed: {e}")
        return 0

    target_hour = _briefing_hour()
    generated = 0
    for user in users:
        user_id = user.get("user_id")
        if not user_id or not is_allowed(user_id):
            continue
        tz = user.get("timezone") or "UTC"
        try:
            now_local = datetime.now(ZoneInfo(tz))
        except (ZoneInfoNotFoundError, Exception):
            now_local = _now()
        if now_local.hour != target_hour:
            continue
        if await _brief_exists_today(user_id, now_local.date().isoformat()):
            continue
        try:
            await generate_brief(user_id, timezone_str=tz)
            generated += 1
        except Exception as e:
            logger.error(f"[briefing] generate failed for {user_id}: {e}", exc_info=True)

    if generated:
        logger.info(f"[briefing] generated {generated} brief(s)")
    return generated
