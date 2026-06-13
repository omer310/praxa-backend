"""AI-powered email classification for email_insights population.

Called by:
  - Nylas webhook handler for real-time inbound email classification
  - Any service that needs to classify and persist an email insight

Writes to `email_insights` with insight_type values the backend understands:
  needs_attention — email requires a reply or action
  awaiting_response — email sent by user with no reply yet (follow-up context)
  no_action — automated, promotional, or informational; not stored
"""
import asyncio
import json as _json
import logging
import os
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

_MODEL = os.getenv("EMAIL_CLASSIFIER_MODEL", "gpt-4o-mini")

_PROMPT = """You are an email priority classifier for an AI executive assistant.

Email:
From: {from_name} <{from_email}>
Subject: {subject}
Snippet: {snippet}

Classify this email. A busy professional relies on you to surface only emails that genuinely need their attention.

SKIP (return no_action) if:
- Automated or transactional (receipts, shipping, security alerts, password resets)
- Promotional or marketing (contains "unsubscribe", offers, newsletters)
- Platform notification (LinkedIn, GitHub, Slack digest, etc. — email is FROM a platform, not the person)
- Informational only (FYI, announcements, no reply needed)

FLAG as needs_attention if:
- A real person is directly asking for a decision, reply, or deliverable
- Active conversation thread where the recipient's response is expected
- Meeting coordination from a person (not a calendar bot)
- Deadline or approval request that requires action

FLAG as awaiting_response if:
- The email shows this person has been waiting on a reply from the recipient for >2 days

Respond with JSON only:
{{"insight_type": "needs_attention" | "awaiting_response" | "no_action", "priority_score": 0.0-1.0, "action_suggested": "Reply confirming..." | null, "reason": "one sentence"}}"""


async def classify_and_store_email_insight(
    user_id: str,
    email_id: str,
    thread_id: str | None,
    from_email: str,
    from_name: str | None,
    subject: str,
    snippet: str,
    received_at: str | None = None,
) -> dict | None:
    """Classify an inbound email and upsert the result into email_insights.

    Returns the classification dict (insight_type, priority_score, etc.) if
    the email warranted storage, or None if it was filtered as no_action.
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        logger.warning("[email_classifier] OPENAI_API_KEY not set; skipping classification")
        return None

    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=openai_key)
        prompt = _PROMPT.format(
            from_name=from_name or from_email,
            from_email=from_email,
            subject=subject or "(no subject)",
            snippet=(snippet or "")[:300],
        )
        response = await client.chat.completions.create(
            model=_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=120,
        )
        analysis = _json.loads(response.choices[0].message.content or "{}")
    except Exception as e:
        logger.error(f"[email_classifier] OpenAI classification failed: {e}")
        return None

    insight_type = analysis.get("insight_type", "no_action")
    if insight_type == "no_action":
        return None

    priority_score = float(analysis.get("priority_score", 0.5))
    if priority_score < 0.4:
        return None

    from .supabase_client import get_supabase_client
    db = get_supabase_client()
    try:
        row = {
            "user_id": user_id,
            "email_id": email_id,
            "thread_id": thread_id,
            "from_email": from_email,
            "from_name": from_name,
            "subject": subject,
            "snippet": snippet,
            "insight_type": insight_type,
            "priority_score": priority_score,
            "action_suggested": analysis.get("action_suggested"),
            "received_at": received_at or datetime.now(timezone.utc).isoformat(),
            "is_addressed": False,
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
        }
        await asyncio.to_thread(
            lambda: db.client.table("email_insights")
            .upsert(row, on_conflict="user_id,email_id")
            .execute()
        )
        logger.info(
            f"[email_classifier] Stored '{insight_type}' insight for user={user_id}: {subject[:60]}"
        )
        return analysis
    except Exception as e:
        logger.error(f"[email_classifier] DB upsert failed for user={user_id}: {e}")
        return None
