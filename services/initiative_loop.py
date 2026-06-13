"""Proactive initiative loop.

Periodically scans each user's attention-worthy inbox (the `email_insights`
rows triage already produced) and, for messages that don't yet have a drafted
reply, composes a short draft and queues it as a `pending_confirmation` action.

Critically: this only ever PROPOSES. Every draft goes through the standard
confirmation gate in action_dispatcher, so nothing is ever sent without the
user approving it (in-app, via push deep-link, or by replying to the SMS/brief).
Output is volume-capped per user so the approval list never floods.
"""
import asyncio
import logging
import os
from datetime import datetime, timezone

from openai import AsyncOpenAI

from .proactive import flag_enabled, is_allowed
from .supabase_client import get_supabase_client

logger = logging.getLogger(__name__)

_MODEL = os.getenv("INITIATIVE_MODEL", "gpt-4o-mini")


def _enabled() -> bool:
    return flag_enabled("PROACTIVE_LOOP_ENABLED", default=True)


def _max_per_day() -> int:
    try:
        return max(1, int(os.getenv("PROACTIVE_MAX_SUGGESTIONS_PER_DAY", "5")))
    except (TypeError, ValueError):
        return 5


def _openai() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


async def _build_ctx(user_id: str):
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
        surface="proactive",
        nylas_api_key=os.getenv("NYLAS_API_KEY"),
    )


async def _attention_emails(user_id: str, limit: int) -> list[dict]:
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
        logger.warning(f"[initiative] attention fetch failed for {user_id}: {e}")
        return []


async def _existing_proactive_drafts(user_id: str) -> tuple[set[str], int]:
    """Return (email_ids already drafted, count of outstanding proactive drafts)."""
    db = get_supabase_client()
    try:
        resp = await asyncio.to_thread(
            lambda: db.client.table("integration_actions")
            .select("payload, status")
            .eq("user_id", user_id)
            .eq("surface", "proactive")
            .eq("action_type", "reply_to_email")
            .in_("status", ["pending_confirmation", "confirmed", "running"])
            .limit(100)
            .execute()
        )
        rows = resp.data or []
        ids = {(r.get("payload") or {}).get("email_id") for r in rows if (r.get("payload") or {}).get("email_id")}
        return ids, len(rows)
    except Exception as e:
        logger.warning(f"[initiative] existing-drafts fetch failed for {user_id}: {e}")
        return set(), 0


async def _draft_reply(email: dict) -> str:
    sender = email.get("from_name") or email.get("from_email") or "there"
    subject = email.get("subject") or "(no subject)"
    snippet = (email.get("snippet") or "")[:400]
    if not os.getenv("OPENAI_API_KEY"):
        return (
            f"Hi {sender.split()[0] if sender else 'there'},\n\nThanks for your email"
            f"{' about ' + subject if subject and subject != '(no subject)' else ''}. "
            "I'll take a look and get back to you shortly.\n\nBest regards"
        )
    prompt = (
        "Draft a concise, professional reply to the email below on the user's behalf. "
        "Plain text, no subject line, no placeholders like [Name] unless unavoidable, 2-4 sentences. "
        "It will be reviewed and edited before sending.\n\n"
        f"From: {sender}\nSubject: {subject}\nPreview: {snippet}"
    )
    try:
        resp = await _openai().chat.completions.create(
            model=_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=220,
        )
        return (resp.choices[0].message.content or "").strip() or "Thanks for your email — I'll follow up shortly."
    except Exception as e:
        logger.warning(f"[initiative] draft LLM failed: {e}")
        return "Thanks for your email — I'll follow up shortly."


async def propose_for_user(user_id: str) -> int:
    """Queue up to the per-day cap of confirm-only reply drafts. Returns count queued."""
    from praxa_core import queue_action

    cap = _max_per_day()
    drafted_ids, outstanding = await _existing_proactive_drafts(user_id)
    budget = cap - outstanding
    if budget <= 0:
        return 0

    emails = await _attention_emails(user_id, limit=cap + len(drafted_ids))
    candidates = [e for e in emails if e.get("email_id") and e["email_id"] not in drafted_ids]
    if not candidates:
        return 0

    ctx = await _build_ctx(user_id)
    queued = 0
    for email in candidates[:budget]:
        reply_body = await _draft_reply(email)
        sender = email.get("from_name") or email.get("from_email") or "a contact"
        subject = email.get("subject") or "(no subject)"
        payload = {
            "email_id": email["email_id"],
            "recipient": email.get("from_email") or "",
            "subject": subject,
            "reply_body": reply_body,
        }
        try:
            await queue_action(
                ctx,
                provider="email",
                action_type="reply_to_email",
                payload=payload,
                summary=f"Reply to {sender} re: {subject[:80]}",
            )
            queued += 1
        except Exception as e:
            logger.error(f"[initiative] queue failed for {user_id}: {e}")

    if queued:
        logger.info(f"[initiative] queued {queued} draft(s) for {user_id}")
        try:
            from .notify_service import notify_user
            await notify_user(
                user_id=user_id,
                event_type="draft_ready",
                title="Praxa drafted replies for you",
                body=f"{queued} email draft{'s' if queued > 1 else ''} ready to review in the app.",
                data={"route": "/email-mode"},
            )
        except Exception as e:
            logger.warning(f"[initiative] SMS/push notify failed for {user_id}: {e}")
    return queued


async def _active_user_ids() -> list[str]:
    """Users worth scanning: those with a connected email grant."""
    db = get_supabase_client()
    try:
        resp = await asyncio.to_thread(
            lambda: db.client.table("nylas_oauth_tokens").select("user_id").eq("integration_type", "email").execute()
        )
        return list({r["user_id"] for r in (resp.data or []) if r.get("user_id")})
    except Exception as e:
        logger.error(f"[initiative] active-user fetch failed: {e}")
        return []


async def run_initiative_loop() -> int:
    """Scheduled entry point. Returns total drafts queued across users."""
    if not _enabled():
        return 0

    user_ids = await _active_user_ids()
    total = 0
    for user_id in user_ids:
        if not is_allowed(user_id):
            continue
        try:
            total += await propose_for_user(user_id)
        except Exception as e:
            logger.error(f"[initiative] propose failed for {user_id}: {e}", exc_info=True)

    if total:
        logger.info(f"[initiative] queued {total} draft(s) across {len(user_ids)} user(s)")
    return total
