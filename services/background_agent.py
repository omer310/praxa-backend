"""Background reasoning agent — Praxa's proactive intelligence engine.

`run_reasoning_pass(user_id)` loads the user's world state snapshot, recent
session summaries, active skills, and notification preferences, then asks
GPT-4o-mini to decide what, if anything, Praxa should do right now:

  notify  — send a push or SMS
  queue_action — create a pending_confirmation integration_action
  call    — trigger an outbound SIP call
  nothing — no action needed

Guard rails:
  - Confidence threshold: only act if confidence >= 0.75
  - Quiet hours: never notify during user's quiet hours
  - Rate limit: at most 1 proactive action per user per 30-min window
    (tracked via user_world_state.last_agent_action_at)
"""
import asyncio
import json as _json
import logging
import os
from datetime import datetime, timezone, timedelta
from typing import Any

logger = logging.getLogger(__name__)

_MODEL = os.getenv("BACKGROUND_AGENT_MODEL", "gpt-4o-mini")
_CONFIDENCE_THRESHOLD = float(os.getenv("BACKGROUND_AGENT_CONFIDENCE", "0.75"))
_WINDOW_MINUTES = 30


_PROMPT = """You are Praxa, an autonomous AI executive assistant. Your job is to decide
whether to proactively help the user right now based on their current world state.

World state:
{world_state}

Recent session context:
{session_context}

User preferences: {preferences}

Decide ONE action or nothing. Options:
  notify   — send the user an alert (push or SMS); use when something truly needs attention
  queue_action — queue a draft action for user approval in the app; use when Praxa should do something
  call     — trigger an outbound phone call; use ONLY if urgent + user unreachable + has phone
  nothing  — no action needed right now

Rules:
  - Do NOT act if urgent_emails=0 AND overdue_tasks=0 AND pending_approvals=0
  - Prefer notify over call; only call if the matter is truly time-sensitive
  - Be conservative: when in doubt, choose nothing
  - Keep reason short and direct — this will be shown to the user

Respond with JSON only:
{{
  "action": "notify" | "queue_action" | "call" | "nothing",
  "reason": "one concise sentence",
  "title": "Short notification title (notify/call only)",
  "body": "Notification body (notify/call only)",
  "confidence": 0.0-1.0
}}"""


async def run_reasoning_pass(user_id: str) -> dict[str, Any] | None:
    """Run one reasoning pass for a user.

    Returns the decision dict if an action was taken, or None if the pass
    was skipped (rate limit, quiet hours, low confidence, or any error).
    """
    from .supabase_client import get_supabase_client
    db = get_supabase_client()

    try:
        world_row, settings_row, sessions_rows, skills_rows = await asyncio.gather(
            _load_world_state(db.client, user_id),
            _load_settings(db.client, user_id),
            _load_recent_sessions(db.client, user_id),
            _load_skills(db.client, user_id),
            return_exceptions=True,
        )

        def _safe(val, default):
            return default if isinstance(val, Exception) else val

        world_row = _safe(world_row, {})
        settings_row = _safe(settings_row, {})
        sessions_rows = _safe(sessions_rows, [])
        skills_rows = _safe(skills_rows, [])

        snapshot = world_row.get("snapshot") or {}
        last_action_at_str = world_row.get("last_agent_action_at")

        if last_action_at_str:
            last_action_at = datetime.fromisoformat(last_action_at_str.replace("Z", "+00:00"))
            if datetime.now(timezone.utc) - last_action_at < timedelta(minutes=_WINDOW_MINUTES):
                logger.debug(f"[bg_agent] Rate-limited for user={user_id} (last_action_at={last_action_at_str})")
                return None

        session_ctx = _format_sessions(sessions_rows)
        skills_ctx = _format_skills(skills_rows)
        prefs = settings_row.get("notification_preferences") or {}

        decision = await _ask_llm(snapshot, session_ctx, prefs)
        if not decision:
            return None

        action = decision.get("action", "nothing")
        confidence = float(decision.get("confidence", 0.0))

        if action == "nothing" or confidence < _CONFIDENCE_THRESHOLD:
            logger.debug(f"[bg_agent] No action for user={user_id} ({action}, conf={confidence:.2f})")
            return None

        from .notify_service import notify_user
        tz_name = settings_row.get("timezone", "UTC")
        if _is_quiet_hours(tz_name):
            logger.debug(f"[bg_agent] Quiet hours for user={user_id} — skipping {action}")
            return None

        await _execute_decision(user_id, decision, db.client, tz_name)
        await _mark_agent_action(db.client, user_id)

        logger.info(f"[bg_agent] Acted for user={user_id}: {action} (conf={confidence:.2f}) — {decision.get('reason', '')}")
        return decision

    except Exception as e:
        logger.error(f"[bg_agent] reasoning pass failed for user={user_id}: {e}", exc_info=True)
        return None


async def _ask_llm(snapshot: dict, session_ctx: str, prefs: dict) -> dict | None:
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        return None
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=openai_key)

        world_text = (
            f"urgent emails needing reply: {snapshot.get('urgent_emails', 0)}\n"
            f"pending in-app approvals: {snapshot.get('pending_approvals', 0)}\n"
            f"overdue tasks: {snapshot.get('overdue_tasks', 0)}\n"
            f"emails awaiting a reply: {snapshot.get('awaiting_replies', 0)}\n"
            f"today's meetings: {', '.join(snapshot.get('todays_meetings', [])) or 'none'}\n"
            f"world state last updated: {snapshot.get('last_updated', 'unknown')}"
        )

        prompt = _PROMPT.format(
            world_state=world_text,
            session_context=session_ctx or "No recent sessions.",
            preferences=_json.dumps(prefs) if prefs else "not configured",
        )

        response = await client.chat.completions.create(
            model=_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=200,
        )
        return _json.loads(response.choices[0].message.content or "{}")
    except Exception as e:
        logger.error(f"[bg_agent] LLM call failed: {e}")
        return None


async def _execute_decision(user_id: str, decision: dict, db, tz_name: str) -> None:
    action = decision.get("action", "nothing")
    title = decision.get("title", "Praxa")
    body = decision.get("body", "")

    from .notify_service import notify_user

    if action == "notify":
        await notify_user(
            user_id=user_id,
            event_type="general",
            title=title,
            body=body,
            data={"source": "background_agent"},
        )
    elif action == "call":
        await _trigger_outbound_call(user_id, decision.get("reason", "background_agent"))
    elif action == "queue_action":
        await _queue_draft_action(user_id, decision, db)


async def _trigger_outbound_call(user_id: str, reason: str) -> None:
    """Trigger an outbound SIP call via the internal endpoint."""
    try:
        import httpx
        backend_url = os.getenv("PRAXA_BACKEND_URL", "http://localhost:8000")
        internal_secret = os.getenv("PRAXA_INTERNAL_SECRET", "")
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(
                f"{backend_url}/internal/trigger-call",
                json={"user_id": user_id, "reason": reason},
                headers={"X-Internal-Secret": internal_secret},
            )
    except Exception as e:
        logger.error(f"[bg_agent] trigger_call failed for user={user_id}: {e}")


async def _queue_draft_action(user_id: str, decision: dict, db) -> None:
    """Queue a generic proactive action for user review."""
    try:
        from datetime import datetime, timezone
        row = {
            "user_id": user_id,
            "provider": "praxa",
            "action_type": "proactive_suggestion",
            "payload": {"reason": decision.get("reason", ""), "source": "background_agent"},
            "summary": decision.get("reason", "Praxa has a suggestion"),
            "status": "pending_confirmation",
            "requires_confirmation": True,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        await asyncio.to_thread(
            lambda: db.table("integration_actions").insert(row).execute()
        )
    except Exception as e:
        logger.error(f"[bg_agent] queue_draft_action failed for user={user_id}: {e}")


def _is_quiet_hours(tz_name: str, quiet_start: int = 22, quiet_end: int = 8) -> bool:
    """Return True if current local time is within quiet hours."""
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(tz_name or "UTC")
        hour = datetime.now(tz).hour
        if quiet_start > quiet_end:
            return hour >= quiet_start or hour < quiet_end
        return quiet_start <= hour < quiet_end
    except Exception:
        return False


async def _mark_agent_action(db, user_id: str) -> None:
    now_iso = datetime.now(timezone.utc).isoformat()
    try:
        await asyncio.to_thread(
            lambda: db.table("user_world_state")
            .upsert(
                {"user_id": user_id, "last_agent_action_at": now_iso, "updated_at": now_iso},
                on_conflict="user_id",
            )
            .execute()
        )
    except Exception as e:
        logger.warning(f"[bg_agent] mark_agent_action failed for user={user_id}: {e}")


async def run_reasoning_pass_all_users() -> None:
    """Run a reasoning pass for every active user. Called by the scheduler."""
    from .supabase_client import get_supabase_client
    db = get_supabase_client()
    try:
        resp = await asyncio.to_thread(
            lambda: db.client.table("users").select("id").eq("ai_enabled", True).execute()
        )
        user_ids = [r["id"] for r in (resp.data or [])]
    except Exception as e:
        logger.error(f"[bg_agent] Could not load user list: {e}")
        return

    logger.info(f"[bg_agent] Running reasoning pass for {len(user_ids)} users")
    for uid in user_ids:
        try:
            await run_reasoning_pass(uid)
        except Exception as e:
            logger.error(f"[bg_agent] Pass failed for user={uid}: {e}")


async def _load_world_state(db, user_id: str) -> dict:
    resp = await asyncio.to_thread(
        lambda: db.table("user_world_state").select("snapshot, last_agent_action_at").eq("user_id", user_id).maybe_single().execute()
    )
    return resp.data or {} if resp else {}


async def _load_settings(db, user_id: str) -> dict:
    resp = await asyncio.to_thread(
        lambda: db.table("user_settings").select("notification_preferences, timezone").eq("user_id", user_id).maybe_single().execute()
    )
    return resp.data or {} if resp else {}


async def _load_recent_sessions(db, user_id: str) -> list:
    resp = await asyncio.to_thread(
        lambda: db.table("session_summaries").select("summary, created_at").eq("user_id", user_id).order("created_at", desc=True).limit(3).execute()
    )
    return resp.data or []


async def _load_skills(db, user_id: str) -> list:
    resp = await asyncio.to_thread(
        lambda: db.table("user_agent_skills").select("name, description").eq("user_id", user_id).eq("status", "active").limit(10).execute()
    )
    return resp.data or []


def _format_sessions(sessions: list) -> str:
    if not sessions:
        return ""
    parts = []
    for s in sessions:
        summary = s.get("summary", "")
        created = s.get("created_at", "")[:10]
        if summary:
            parts.append(f"[{created}] {summary[:200]}")
    return "\n".join(parts)


def _format_skills(skills: list) -> str:
    if not skills:
        return ""
    return ", ".join(s.get("name", "") for s in skills if s.get("name"))
