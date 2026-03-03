"""
Shared memory layer for Praxa agents.
Extracts user facts and session summaries from conversations,
stores them in Supabase, and loads them at session start.
"""

import json
import logging
import os
from datetime import datetime
from typing import Optional

from openai import AsyncOpenAI
from supabase import Client

logger = logging.getLogger(__name__)

openai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")


def _get_supabase() -> Client:
    from supabase import create_client
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


async def _get_embedding(text: str) -> Optional[list[float]]:
    try:
        response = await openai.embeddings.create(
            model="text-embedding-3-small",
            input=text[:8000],
        )
        return response.data[0].embedding
    except Exception as e:
        logger.warning(f"[Memory] Embedding failed: {e}")
        return None


async def extract_and_store_session_memory(
    user_id: str,
    surface: str,
    transcript: list | str,
    summary: str,
    duration: int = 0,
    session_id: Optional[str] = None,
) -> None:
    """
    After a session ends, extract user facts and store the session summary.

    Args:
        user_id: Supabase auth user ID
        surface: 'phone' | 'voice' | 'text'
        transcript: List of {speaker, text} dicts or a plain string
        summary: Summary of the session (may already be generated)
        duration: Session duration in seconds
        session_id: Optional identifier for this session
    """
    try:
        transcript_text = (
            "\n".join(f"{m.get('speaker', 'unknown')}: {m.get('text', '')}" for m in transcript)
            if isinstance(transcript, list)
            else str(transcript)
        )

        if not transcript_text.strip():
            logger.info(f"[Memory] Empty transcript for user {user_id}, skipping")
            return

        extraction_prompt = f"""You are analyzing a conversation between a user and their AI assistant Praxa.
Extract useful, durable facts about the user from this conversation.
Return a JSON object with two keys:
- "facts": array of {{key: string, value: string, confidence: number (0-1)}} — persistent user preferences, habits, goals, constraints
- "summary": 2-3 sentence summary of what happened in this conversation

Conversation transcript:
{transcript_text[:6000]}

Existing summary to refine (if provided): {summary[:500] if summary else 'None'}

Return ONLY valid JSON, no markdown."""

        response = await openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": extraction_prompt}],
            temperature=0.3,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content or "{}"
        parsed = json.loads(raw)

        extracted_summary = parsed.get("summary", summary or "Session completed.")
        facts = parsed.get("facts", [])

        db = _get_supabase()

        summary_embedding = await _get_embedding(extracted_summary)
        summary_row: dict = {
            "user_id": user_id,
            "surface": surface,
            "summary": extracted_summary,
            "key_facts": json.dumps(facts),
            "duration_seconds": duration,
            "created_at": datetime.utcnow().isoformat(),
        }
        if session_id:
            summary_row["session_id"] = session_id
        if summary_embedding:
            summary_row["embedding"] = summary_embedding

        db.table("session_summaries").insert(summary_row).execute()
        logger.info(f"[Memory] Stored session summary for user {user_id} ({surface})")

        for fact in facts[:20]:
            key = str(fact.get("key", "")).strip()
            value = str(fact.get("value", "")).strip()
            confidence = float(fact.get("confidence", 0.7))
            if not key or not value:
                continue

            fact_embedding = await _get_embedding(f"{key}: {value}")
            fact_row: dict = {
                "user_id": user_id,
                "fact_key": key,
                "fact_value": value,
                "source": surface,
                "confidence": confidence,
                "last_confirmed_at": datetime.utcnow().isoformat(),
            }
            if fact_embedding:
                fact_row["embedding"] = fact_embedding

            db.table("user_facts").upsert(
                fact_row, on_conflict="user_id,fact_key"
            ).execute()

        logger.info(f"[Memory] Stored {len(facts)} facts for user {user_id}")

    except Exception as e:
        logger.error(f"[Memory] extract_and_store_session_memory failed: {e}", exc_info=True)


async def load_session_context(user_id: str) -> str:
    """
    Load relevant memory context for the start of a session.

    Returns a formatted string ready to inject into a system prompt.
    """
    try:
        db = _get_supabase()

        summaries_resp = (
            db.table("session_summaries")
            .select("summary, surface, created_at")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .limit(5)
            .execute()
        )

        facts_resp = (
            db.table("user_facts")
            .select("fact_key, fact_value, confidence")
            .eq("user_id", user_id)
            .order("last_confirmed_at", desc=True)
            .limit(20)
            .execute()
        )

        summaries = summaries_resp.data or []
        facts = facts_resp.data or []

        if not summaries and not facts:
            return ""

        lines = ["## Memory Context\n"]

        if facts:
            lines.append("### What I know about you")
            for f in facts:
                lines.append(f"- {f['fact_key']}: {f['fact_value']}")
            lines.append("")

        if summaries:
            lines.append("### Recent session history")
            for s in summaries:
                surface_label = {"phone": "phone call", "voice": "voice chat", "text": "text chat"}.get(s["surface"], s["surface"])
                lines.append(f"- [{surface_label}] {s['summary']}")
            lines.append("")

        return "\n".join(lines)

    except Exception as e:
        logger.error(f"[Memory] load_session_context failed: {e}", exc_info=True)
        return ""


async def consolidate_memories(user_id: str) -> None:
    """
    Merge redundant user_facts entries and prune very old/low-confidence facts.
    Runs nightly via scheduler.
    """
    try:
        db = _get_supabase()

        resp = (
            db.table("user_facts")
            .select("id, fact_key, fact_value, confidence, last_confirmed_at")
            .eq("user_id", user_id)
            .execute()
        )
        facts = resp.data or []

        if len(facts) < 2:
            return

        facts_text = "\n".join(f"{f['fact_key']}: {f['fact_value']} (confidence: {f['confidence']})" for f in facts)
        consolidation_prompt = f"""You are reviewing a list of user facts extracted from conversations.
Identify and merge duplicates/contradictions, keeping the most accurate version.
Return a JSON array of {{key, value, confidence}} objects, with redundancies removed.
Keep at most 30 facts. Remove facts with confidence < 0.4.

Facts:
{facts_text[:4000]}

Return ONLY valid JSON array."""

        response = await openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": consolidation_prompt}],
            temperature=0.2,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content or "[]"
        try:
            parsed = json.loads(raw)
            consolidated = parsed if isinstance(parsed, list) else parsed.get("facts", [])
        except Exception:
            logger.warning("[Memory] Could not parse consolidation response")
            return

        db.table("user_facts").delete().eq("user_id", user_id).execute()

        for fact in consolidated:
            key = str(fact.get("key", "")).strip()
            value = str(fact.get("value", "")).strip()
            confidence = float(fact.get("confidence", 0.7))
            if not key or not value:
                continue

            fact_embedding = await _get_embedding(f"{key}: {value}")
            fact_row: dict = {
                "user_id": user_id,
                "fact_key": key,
                "fact_value": value,
                "source": "consolidated",
                "confidence": confidence,
                "last_confirmed_at": datetime.utcnow().isoformat(),
            }
            if fact_embedding:
                fact_row["embedding"] = fact_embedding

            db.table("user_facts").insert(fact_row).execute()

        logger.info(f"[Memory] Consolidated facts for user {user_id}: {len(facts)} → {len(consolidated)}")

    except Exception as e:
        logger.error(f"[Memory] consolidate_memories failed: {e}", exc_info=True)


async def consolidate_all_users_memories() -> None:
    """Run memory consolidation for all active users. Called by nightly scheduler."""
    try:
        from services.supabase_client import get_supabase_client
        db_client = get_supabase_client()

        resp = db_client.client.table("user_facts").select("user_id").execute()
        user_ids = list({row["user_id"] for row in (resp.data or [])})

        logger.info(f"[Memory] Consolidating memories for {len(user_ids)} users")
        for uid in user_ids:
            await consolidate_memories(uid)

    except Exception as e:
        logger.error(f"[Memory] consolidate_all_users_memories failed: {e}", exc_info=True)
