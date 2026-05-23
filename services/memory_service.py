"""
Shared memory layer for Praxa agents.
Extracts user facts and session summaries from conversations,
stores them in Supabase, and loads them at session start.
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone

_UTC = timezone.utc
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
            model="gpt-5.4-mini",
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
            "key_facts": facts,
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


async def load_session_context(user_id: str, query_text: str | None = None) -> str:
    """
    Load tiered memory context for the start of a session.

    Three-tier loading strategy:
    - WARM (always): core profile from user_ai_memory + latest compressed history
    - HOT  (always): last 3 non-archived raw session summaries
    - COLD (vector):  archived summaries retrieved by semantic similarity to session context
    - FACTS (vector): top relevant user facts, falling back to recency

    Falls back to recency-based loading for facts/summaries if embedding fails.
    """
    try:
        db = _get_supabase()

        effective_query = query_text or "user preferences recent activity current tasks behavioral patterns"
        query_embedding = await _get_embedding(effective_query)

        # --- WARM tier: core profile + compressed history (always, no retrieval) ---
        profile_resp = await asyncio.to_thread(
            lambda: db.table("user_ai_memory")
            .select("value")
            .eq("user_id", user_id)
            .eq("memory_type", "core_profile")
            .eq("key", "profile")
            .maybe_single()
            .execute()
        )
        core_profile_text: str | None = None
        if profile_resp.data:
            val = profile_resp.data.get("value")
            if isinstance(val, dict):
                core_profile_text = val.get("text")
            elif isinstance(val, str):
                core_profile_text = val

        compressed_resp = await asyncio.to_thread(
            lambda: db.table("session_summaries")
            .select("summary, created_at")
            .eq("user_id", user_id)
            .eq("surface", "compressed")
            .eq("is_archived", False)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        compressed = (compressed_resp.data or [None])[0]

        # --- HOT tier: last 3 recent non-archived raw summaries (always) ---
        recent_resp = await asyncio.to_thread(
            lambda: db.table("session_summaries")
            .select("summary, surface, created_at")
            .eq("user_id", user_id)
            .neq("surface", "compressed")
            .eq("is_archived", False)
            .order("created_at", desc=True)
            .limit(3)
            .execute()
        )
        recent_summaries = recent_resp.data or []

        # --- FACTS + COLD tier: vector search when embedding available ---
        facts: list[dict] = []
        archived_summaries: list[dict] = []
        used_vector_search = False

        if query_embedding:
            try:
                facts_rpc_task = asyncio.to_thread(
                    lambda: db.rpc(
                        "match_user_facts",
                        {"p_user_id": user_id, "p_query_embedding": query_embedding, "p_match_count": 15, "p_match_threshold": 0.3},
                    ).execute()
                )
                archived_rpc_task = asyncio.to_thread(
                    lambda: db.rpc(
                        "match_archived_session_summaries",
                        {"p_user_id": user_id, "p_query_embedding": query_embedding, "p_match_count": 3, "p_match_threshold": 0.35},
                    ).execute()
                )
                rpc_facts, rpc_archived = await asyncio.gather(
                    facts_rpc_task, archived_rpc_task, return_exceptions=True
                )
                if not isinstance(rpc_facts, Exception):
                    facts = rpc_facts.data or []
                if not isinstance(rpc_archived, Exception):
                    archived_summaries = rpc_archived.data or []
                used_vector_search = True
            except Exception as rpc_err:
                logger.warning(f"[Memory] Vector RPC failed, falling back to recency for facts: {rpc_err}")

        if not used_vector_search or not facts:
            facts_resp = await asyncio.to_thread(
                lambda: db.table("user_facts")
                .select("fact_key, fact_value, confidence")
                .eq("user_id", user_id)
                .order("last_confirmed_at", desc=True)
                .limit(20)
                .execute()
            )
            facts = facts_resp.data or []

        if not core_profile_text and not facts and not recent_summaries and not compressed:
            return ""

        lines = ["## Memory Context\n"]

        if core_profile_text:
            lines.append("### Core Profile (always loaded)")
            lines.append(core_profile_text)
            lines.append("")

        if facts:
            lines.append("### What I know about you")
            for f in facts:
                lines.append(f"- {f['fact_key']}: {f['fact_value']}")
            lines.append("")

        if compressed:
            lines.append("### Long-term history (compressed)")
            lines.append(compressed["summary"])
            lines.append("")

        if recent_summaries:
            lines.append("### Recent sessions")
            for s in recent_summaries:
                surface_label = {"phone": "phone call", "voice": "voice chat", "text": "text chat"}.get(
                    s.get("surface", ""), s.get("surface", "")
                )
                lines.append(f"- [{surface_label}] {s['summary']}")
            lines.append("")

        if archived_summaries:
            lines.append("### Relevant past sessions")
            for s in archived_summaries:
                lines.append(f"- [archived {s['created_at'][:10]}] {s['summary']}")
            lines.append("")

        return "\n".join(lines)

    except Exception as e:
        logger.error(f"[Memory] load_session_context failed: {e}", exc_info=True)
        return ""


async def extract_skills_from_session(
    user_id: str,
    transcript: list | str,
    surface: str,
) -> None:
    """
    Extract behavioral skills from a session transcript and upsert them.
    Only creates skills when clear, durable behavioral patterns are observed.
    """
    try:
        transcript_text = (
            "\n".join(
                f"{m.get('role', m.get('speaker', 'unknown'))}: {m.get('content', m.get('text', ''))}"
                for m in transcript
            )
            if isinstance(transcript, list)
            else str(transcript)
        )

        if len(transcript_text.strip()) < 200:
            return

        extraction_prompt = f"""You are analyzing a conversation to extract behavioral skill modules.
A "skill" is a durable, reusable behavioral pattern worth capturing — like a communication preference, a workflow habit, or a recurring ritual. Not every conversation contains a skill worth extracting.

Return a JSON object with:
- "skills": array of proposed skills, each with:
  - name: string (3-5 words)
  - slug: string (snake_case)
  - description: string (one sentence for the user to understand)
  - content: string (2-4 sentences of instructions for an AI assistant)
  - category: string ("communication" | "workflow" | "personal" | "productivity")

Examples of good skills:
- name: "Email Communication Style", content: "Prefers concise bullet-point emails under 150 words. Always lead with the ask. Avoid pleasantries."
- name: "Weekly Review Ritual", content: "Every Sunday evening, help with reviewing the past week's tasks and planning the next sprint."

Only extract if strong, clear evidence exists. If nothing strong emerges, return {{"skills": []}}.

Conversation:
{transcript_text[:5000]}

Return ONLY valid JSON."""

        response = await openai.chat.completions.create(
            model="gpt-5.4-mini",
            messages=[{"role": "user", "content": extraction_prompt}],
            temperature=0.3,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content or "{}"
        parsed = json.loads(raw)
        skills = parsed.get("skills", [])

        if not skills:
            return

        db = _get_supabase()
        stored = 0
        for skill in skills[:3]:
            name = str(skill.get("name", "")).strip()
            slug = str(skill.get("slug", "")).strip().replace(" ", "_").lower()
            content = str(skill.get("content", "")).strip()
            description = str(skill.get("description", "")).strip()
            category = str(skill.get("category", "")).strip() or None

            if not name or not slug or not content:
                continue

            existing = db.table("user_agent_skills").select("source").eq(
                "user_id", user_id
            ).eq("slug", slug).eq("status", "active").maybeSingle().execute()
            if existing.data and existing.data.get("source") == "manual":
                continue

            skill_row = {
                "user_id": user_id,
                "name": name,
                "slug": slug,
                "description": description,
                "content": content,
                "category": category,
                "source": "prompted",
                "status": "active",
                "confidence": 0.8,
            }
            db.table("user_agent_skills").upsert(
                skill_row, on_conflict="user_id,slug"
            ).execute()
            stored += 1

        if stored:
            logger.info(f"[Skills] Extracted {stored} skills from {surface} session for user {user_id}")

    except Exception as e:
        logger.error(f"[Skills] extract_skills_from_session failed: {e}", exc_info=True)


def _compute_decayed_confidence(confidence: float, last_confirmed_at: str | None) -> float:
    """
    Apply Ebbinghaus-style monthly decay to a fact's confidence.
    Decay factor: 0.95 per month (5% monthly reduction).
    Skills and procedures use λ=0 (no decay) — handled in skill_service.
    """
    if not last_confirmed_at:
        return confidence
    try:
        last_dt = datetime.fromisoformat(last_confirmed_at.replace("Z", "+00:00"))
        if last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=_UTC)
        now = datetime.now(_UTC)
        months_elapsed = max(0.0, (now - last_dt).days / 30.0)
        return round(confidence * (0.95 ** months_elapsed), 4)
    except Exception:
        return confidence


async def consolidate_memories(user_id: str) -> None:
    """
    Merge redundant user_facts, apply time-based confidence decay, and prune
    low-signal facts. Non-destructive: upserts survivors and only deletes the
    keys GPT explicitly removed, so no data is lost on an LLM failure.
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

        now = datetime.now(_UTC)
        facts_lines = []
        for f in facts:
            last_confirmed = f.get("last_confirmed_at")
            try:
                last_dt = datetime.fromisoformat((last_confirmed or "").replace("Z", "+00:00"))
                if last_dt.tzinfo is None:
                    last_dt = last_dt.replace(tzinfo=_UTC)
                days_ago = (now - last_dt).days
                staleness_label = f"{days_ago}d ago"
            except Exception:
                staleness_label = "unknown age"

            decayed = _compute_decayed_confidence(
                float(f.get("confidence", 0.7)), last_confirmed
            )
            facts_lines.append(
                f"{f['fact_key']}: {f['fact_value']} "
                f"(confidence: {decayed:.2f}, last seen: {staleness_label})"
            )

        facts_text = "\n".join(facts_lines)
        consolidation_prompt = f"""You are reviewing a list of user facts extracted from AI conversations.
Each fact shows its current confidence score (already decay-adjusted) and how long ago it was last confirmed.

Your tasks:
1. Merge duplicate or near-duplicate facts — keep the most accurate, up-to-date version.
2. Resolve contradictions — prefer the higher-confidence fact; note that older facts may be outdated.
3. Remove facts with a decayed confidence below 0.35, or facts unseen for more than 180 days with confidence below 0.5.
4. Keep at most 30 facts total.

Return a JSON object with a "facts" array where each item has:
  - key: string (concise snake_case label)
  - value: string (the fact value)
  - confidence: number (0.0–1.0, your best estimate after merging)

If nothing needs changing, return the cleaned-up list as-is. NEVER return an empty facts array unless every single fact should be removed.

Facts:
{facts_text[:4000]}

Return ONLY valid JSON."""

        response = await openai.chat.completions.create(
            model="gpt-5.4-mini",
            messages=[{"role": "user", "content": consolidation_prompt}],
            temperature=0.2,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content or "{}"
        try:
            parsed = json.loads(raw)
            consolidated = parsed.get("facts", parsed) if isinstance(parsed, dict) else parsed
            if not isinstance(consolidated, list):
                consolidated = []
        except Exception:
            logger.warning("[Memory] Could not parse consolidation response — keeping existing facts")
            return

        if not consolidated:
            logger.warning("[Memory] Consolidation returned empty list — aborting to avoid data loss")
            return

        if len(consolidated) < max(1, len(facts) * 0.2):
            logger.warning(
                f"[Memory] Consolidation shrank facts from {len(facts)} to {len(consolidated)} "
                f"(>80% reduction) — aborting as a safety guard"
            )
            return

        consolidated_keys = set()
        for fact in consolidated:
            key = str(fact.get("key", "")).strip()
            value = str(fact.get("value", "")).strip()
            confidence = float(fact.get("confidence", 0.7))
            if not key or not value:
                continue
            consolidated_keys.add(key)
            fact_embedding = await _get_embedding(f"{key}: {value}")
            fact_row: dict = {
                "user_id": user_id,
                "fact_key": key,
                "fact_value": value,
                "source": "consolidated",
                "confidence": confidence,
                "last_confirmed_at": datetime.now(_UTC).isoformat(),
            }
            if fact_embedding:
                fact_row["embedding"] = fact_embedding
            db.table("user_facts").upsert(fact_row, on_conflict="user_id,fact_key").execute()

        removed_keys = [f["fact_key"] for f in facts if f["fact_key"] not in consolidated_keys]
        for key in removed_keys:
            db.table("user_facts").delete().eq("user_id", user_id).eq("fact_key", key).execute()

        logger.info(
            f"[Memory] Consolidated facts for user {user_id}: "
            f"{len(facts)} → {len(consolidated_keys)} kept, {len(removed_keys)} removed"
        )

    except Exception as e:
        logger.error(f"[Memory] consolidate_memories failed: {e}", exc_info=True)


async def compress_session_summaries(user_id: str) -> None:
    """
    When a user has more than 10 raw session summaries, compress the oldest ones
    (everything beyond the 5 most recent) into a single 'historical profile' row
    with surface='compressed', then archive the raw rows that were compressed.

    Archived rows are marked is_archived=True rather than deleted, so the full
    history is preserved and retrievable via vector search.
    Loading logic in load_session_context picks up the compressed row as context,
    keeping the injected prompt small regardless of how many sessions a user has had.
    Runs nightly via scheduler (after memory consolidation).
    """
    try:
        db = _get_supabase()

        raw_resp = (
            db.table("session_summaries")
            .select("id, summary, surface, created_at")
            .eq("user_id", user_id)
            .neq("surface", "compressed")
            .eq("is_archived", False)
            .order("created_at", desc=True)
            .execute()
        )
        raw_summaries = raw_resp.data or []

        if len(raw_summaries) <= 10:
            return

        recent = raw_summaries[:5]
        to_compress = raw_summaries[5:]

        existing_compressed_resp = (
            db.table("session_summaries")
            .select("id, summary")
            .eq("user_id", user_id)
            .eq("surface", "compressed")
            .eq("is_archived", False)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        existing_compressed = (existing_compressed_resp.data or [None])[0]

        older_text_parts = []
        if existing_compressed:
            older_text_parts.append(f"[Previously compressed history]\n{existing_compressed['summary']}")
        for s in reversed(to_compress):
            surface_label = {"phone": "phone call", "voice": "voice chat", "text": "text chat"}.get(
                s.get("surface", ""), s.get("surface", "")
            )
            older_text_parts.append(f"[{surface_label} on {s['created_at'][:10]}] {s['summary']}")

        older_text = "\n".join(older_text_parts)

        compression_prompt = f"""You are summarizing older session history for a personal AI assistant called Praxa.
The following are past session summaries (oldest first). Compress them into a single concise paragraph (4–6 sentences) that captures:
- The user's recurring themes, habits, and patterns
- Important decisions or milestones mentioned
- Any long-term context that would help Praxa understand this person

Write in third person about the user. Be specific, not generic. Omit one-off events that have no lasting relevance.

Session history to compress:
{older_text[:5000]}

Return only the compressed paragraph, no headers or labels."""

        response = await openai.chat.completions.create(
            model="gpt-5.4-mini",
            messages=[{"role": "user", "content": compression_prompt}],
            temperature=0.3,
        )

        compressed_text = (response.choices[0].message.content or "").strip()
        if not compressed_text:
            logger.warning(f"[Memory] Compression returned empty text for user {user_id} — skipping")
            return

        compressed_embedding = await _get_embedding(compressed_text)
        compressed_row: dict = {
            "user_id": user_id,
            "surface": "compressed",
            "summary": compressed_text,
            "key_facts": [],
            "duration_seconds": 0,
            "created_at": datetime.now(_UTC).isoformat(),
        }
        if compressed_embedding:
            compressed_row["embedding"] = compressed_embedding

        db.table("session_summaries").insert(compressed_row).execute()

        ids_to_archive = [s["id"] for s in to_compress]
        if existing_compressed:
            ids_to_archive.append(existing_compressed["id"])

        for row_id in ids_to_archive:
            db.table("session_summaries").update({"is_archived": True}).eq("id", row_id).execute()

        logger.info(
            f"[Memory] Compressed {len(to_compress)} summaries for user {user_id} "
            f"(+1 prior compressed → 1 new compressed, {len(recent)} raw kept, {len(ids_to_archive)} archived)"
        )

    except Exception as e:
        logger.error(f"[Memory] compress_session_summaries failed for user {user_id}: {e}", exc_info=True)


async def generate_core_profile(user_id: str) -> None:
    """
    Synthesize a compact, always-loaded core profile paragraph (~300 tokens) for a user
    and upsert it into user_ai_memory as memory_type='core_profile', key='profile'.

    The core profile is the WARM tier of the memory stack — stable traits, top preferences,
    and behavioral patterns distilled from high-confidence facts and compressed history.
    It is always injected at session start without any retrieval step, so it must be concise.
    Runs nightly after memory consolidation and summary compression.
    """
    try:
        db = _get_supabase()

        facts_resp = (
            db.table("user_facts")
            .select("fact_key, fact_value, confidence")
            .eq("user_id", user_id)
            .gte("confidence", 0.6)
            .order("confidence", desc=True)
            .limit(20)
            .execute()
        )
        facts = facts_resp.data or []

        if not facts:
            return

        compressed_resp = (
            db.table("session_summaries")
            .select("summary")
            .eq("user_id", user_id)
            .eq("surface", "compressed")
            .eq("is_archived", False)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        compressed_row = (compressed_resp.data or [None])[0]

        facts_text = "\n".join(f"- {f['fact_key']}: {f['fact_value']}" for f in facts)
        history_text = f"\n\nLong-term history summary:\n{compressed_row['summary']}" if compressed_row else ""

        profile_prompt = f"""You are building a compact core profile for a personal AI assistant called Praxa.
The profile captures who this user is — their stable traits, key preferences, work context, and behavioral patterns.
It will be injected at the start of every session so it must be dense and useful, under 250 words.

Write a single paragraph in second person (addressing the AI, not the user): "The user is..." or "They prefer..."
Be specific — names, roles, habits — not generic. Omit anything that sounds like a one-off event.

Source material:
{facts_text}{history_text}

Return only the profile paragraph, no headers, labels, or preamble."""

        response = await openai.chat.completions.create(
            model="gpt-5.4-mini",
            messages=[{"role": "user", "content": profile_prompt}],
            temperature=0.3,
        )

        profile_text = (response.choices[0].message.content or "").strip()
        if not profile_text:
            logger.warning(f"[Memory] Core profile generation returned empty text for user {user_id}")
            return

        db.table("user_ai_memory").upsert(
            {
                "user_id": user_id,
                "memory_type": "core_profile",
                "key": "profile",
                "value": {"text": profile_text},
                "confidence": 1.0,
                "updated_at": datetime.now(_UTC).isoformat(),
                "last_used_at": datetime.now(_UTC).isoformat(),
            },
            on_conflict="user_id,memory_type,key",
        ).execute()

        logger.info(f"[Memory] Generated core profile for user {user_id} ({len(profile_text)} chars)")

    except Exception as e:
        logger.error(f"[Memory] generate_core_profile failed for user {user_id}: {e}", exc_info=True)


async def consolidate_all_users_memories() -> None:
    """Run full nightly memory pipeline for all active users. Called by scheduler."""
    try:
        from services.supabase_client import get_supabase_client
        db_client = get_supabase_client()

        resp = db_client.client.table("user_facts").select("user_id").execute()
        user_ids = list({row["user_id"] for row in (resp.data or [])})

        logger.info(f"[Memory] Running nightly pipeline for {len(user_ids)} users")
        for uid in user_ids:
            await consolidate_memories(uid)
            await compress_session_summaries(uid)
            await generate_core_profile(uid)

    except Exception as e:
        logger.error(f"[Memory] consolidate_all_users_memories failed: {e}", exc_info=True)
