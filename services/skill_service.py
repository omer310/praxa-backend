"""Agent Skills service — propose skills from user facts nightly."""

import json
import logging
import os

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

openai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")


def _get_supabase():
    from supabase import create_client
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


async def extract_skills_from_session(
    user_id: str,
    transcript: list | str,
    surface: str,
) -> None:
    """
    Extract behavioral skills from a session transcript and upsert them.
    Imported from memory_service to avoid circular imports; re-exported here
    for callers that prefer importing from skill_service.
    """
    from services.memory_service import extract_skills_from_session as _extract
    await _extract(user_id=user_id, transcript=transcript, surface=surface)


async def consolidate_skills(user_id: str) -> None:
    """
    Merge and deduplicate active agent skills for a user.

    When a user has more than 5 active skills, run a GPT pass to:
    - Merge skills with overlapping intent into a single better-worded version
    - Absorb narrow skills that are subsets of broader ones
    - Keep skills that are genuinely distinct unchanged

    Merged/absorbed originals are soft-deleted (status='archived') so they remain
    auditable and are excluded from all future loads. The surviving consolidated
    skills are upserted by slug.
    Skills have λ=0 (no decay) — a learned behavioral rule stays valid until
    explicitly contradicted, not just because time has passed.
    Runs nightly via scheduler after skill proposals.
    """
    try:
        db = _get_supabase()

        resp = (
            db.table("user_agent_skills")
            .select("id, name, slug, content, category, source")
            .eq("user_id", user_id)
            .eq("status", "active")
            .order("created_at")
            .execute()
        )
        skills = resp.data or []

        if len(skills) <= 5:
            return

        skills_text = "\n".join(
            f"[{i+1}] slug={s['slug']} | category={s.get('category', 'unknown')} | name={s['name']}\n{s['content']}"
            for i, s in enumerate(skills)
        )

        consolidation_prompt = f"""You are reviewing a set of behavioral "Agent Skills" for a personal AI assistant.
Each skill is an instruction module injected into the AI's system prompt.

Your task:
- Identify skills with overlapping or near-duplicate intent (>30% conceptual overlap).
- Merge overlapping skills into a single, better-worded skill that captures the combined intent.
- Absorb narrow skills that are subsets of a broader one into the broader skill.
- Leave genuinely distinct skills unchanged.
- Do NOT merge skills from different categories unless they are truly redundant.

Return a JSON object with:
- "consolidated": array of surviving skills after merging, each with:
    - slug: string (reuse an existing slug; prefer the most descriptive one)
    - name: string (3-5 words; reuse or improve)
    - content: string (2-4 sentences of AI instructions; merged/improved content)
    - category: string ("communication" | "workflow" | "personal" | "productivity")
    - merged_slugs: array of slugs that were absorbed into this skill (empty if unchanged)

Only include skills in the output that SURVIVE (either unchanged or as the merged result).
Skills whose slugs appear only in merged_slugs of another skill are being removed.

Agent Skills:
{skills_text[:4000]}

Return ONLY valid JSON."""

        response = await openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": consolidation_prompt}],
            temperature=0.2,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content or "{}"
        parsed = json.loads(raw)
        consolidated = parsed.get("consolidated", [])

        if not consolidated:
            logger.warning(f"[Skills] Consolidation returned empty list for user {user_id} — skipping")
            return

        safety_ratio = len(consolidated) / len(skills)
        if safety_ratio < 0.3:
            logger.warning(
                f"[Skills] Consolidation would remove >70% of skills ({len(skills)} → {len(consolidated)}) "
                f"for user {user_id} — skipping as safety guard"
            )
            return

        all_absorbed_slugs: set[str] = set()
        for skill in consolidated:
            for absorbed in skill.get("merged_slugs", []):
                all_absorbed_slugs.add(str(absorbed))

        for skill in consolidated:
            slug = str(skill.get("slug", "")).strip()
            name = str(skill.get("name", "")).strip()
            content = str(skill.get("content", "")).strip()
            category = str(skill.get("category", "")).strip() or None

            if not slug or not name or not content:
                continue

            db.table("user_agent_skills").upsert(
                {
                    "user_id": user_id,
                    "slug": slug,
                    "name": name,
                    "content": content,
                    "category": category,
                    "status": "active",
                    "source": "auto_proposed",
                    "confidence": 1.0,
                },
                on_conflict="user_id,slug",
            ).execute()

        for absorbed_slug in all_absorbed_slugs:
            db.table("user_agent_skills").update({"status": "archived"}).eq(
                "user_id", user_id
            ).eq("slug", absorbed_slug).eq("status", "active").execute()

        removed_count = len(all_absorbed_slugs)
        logger.info(
            f"[Skills] Consolidated skills for user {user_id}: "
            f"{len(skills)} → {len(consolidated)} active, {removed_count} archived/absorbed"
        )

    except Exception as e:
        logger.error(f"[Skills] consolidate_skills failed for user {user_id}: {e}", exc_info=True)


async def consolidate_skills_for_all_users() -> None:
    """Run skill consolidation for all active users. Called by nightly scheduler."""
    try:
        from services.supabase_client import get_supabase_client
        db_client = get_supabase_client()

        resp = db_client.client.table("user_agent_skills").select("user_id").eq("status", "active").execute()
        user_ids = list({row["user_id"] for row in (resp.data or [])})

        logger.info(f"[Skills] Consolidating skills for {len(user_ids)} users")
        for uid in user_ids:
            await consolidate_skills(uid)

    except Exception as e:
        logger.error(f"[Skills] consolidate_skills_for_all_users failed: {e}", exc_info=True)


async def propose_skills_from_facts(user_id: str) -> None:
    """
    Cluster a user's facts into agent skills and activate them automatically.
    """
    try:
        db = _get_supabase()

        facts_resp = (
            db.table("user_facts")
            .select("id, fact_key, fact_value, confidence")
            .eq("user_id", user_id)
            .order("last_confirmed_at", desc=True)
            .limit(30)
            .execute()
        )
        facts = facts_resp.data or []

        if len(facts) < 3:
            return

        facts_text = "\n".join(
            f"[{f['id']}] {f['fact_key']}: {f['fact_value']}" for f in facts
        )

        prompt = f"""You are analyzing a list of facts about a user to identify behavioral patterns that could become reusable "Agent Skills" — named instruction modules injected into an AI assistant's system prompt.

A skill should:
- Represent a durable preference, workflow, ritual, or communication style
- Be actionable (an AI can use it to behave differently)
- Be broad enough to apply across multiple conversations

Return a JSON object with:
- "skills": array of up to 3 proposed skills, each with:
  - name: string (3-5 words)
  - slug: string (snake_case)
  - description: string (one sentence the user will read)
  - content: string (2-4 sentences of instructions for an AI assistant)
  - category: string ("communication" | "workflow" | "personal" | "productivity")
  - source_fact_ids: array of fact IDs from the input that support this skill

Only propose skills when there is clear, consistent evidence across multiple facts. If nothing strong emerges, return {{"skills": []}}.

User facts:
{facts_text[:3000]}

Return ONLY valid JSON."""

        response = await openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content or "{}"
        parsed = json.loads(raw)
        proposed = parsed.get("skills", [])

        if not proposed:
            return

        stored = 0
        for skill in proposed[:3]:
            name = str(skill.get("name", "")).strip()
            slug = str(skill.get("slug", "")).strip().replace(" ", "_").lower()
            content = str(skill.get("content", "")).strip()
            description = str(skill.get("description", "")).strip()
            category = str(skill.get("category", "")).strip() or None
            source_fact_ids = skill.get("source_fact_ids", [])

            if not name or not slug or not content:
                continue

            existing_active = (
                db.table("user_agent_skills")
                .select("id")
                .eq("user_id", user_id)
                .eq("slug", slug)
                .eq("status", "active")
                .execute()
            )
            if existing_active.data:
                continue

            skill_row = {
                "user_id": user_id,
                "name": name,
                "slug": slug,
                "description": description,
                "content": content,
                "category": category,
                "source": "auto_proposed",
                "status": "active",
                "confidence": 0.7,
                "source_fact_ids": source_fact_ids if isinstance(source_fact_ids, list) else [],
            }
            db.table("user_agent_skills").insert(skill_row).execute()
            stored += 1

        if stored:
            logger.info(f"[Skills] Proposed {stored} skills for user {user_id}")

    except Exception as e:
        logger.error(f"[Skills] propose_skills_from_facts failed for user {user_id}: {e}", exc_info=True)


async def propose_skills_for_all_users() -> None:
    """Run skill proposal for all active users. Called by nightly scheduler."""
    try:
        from services.supabase_client import get_supabase_client
        db_client = get_supabase_client()

        resp = db_client.client.table("user_facts").select("user_id").execute()
        user_ids = list({row["user_id"] for row in (resp.data or [])})

        logger.info(f"[Skills] Proposing skills for {len(user_ids)} users")
        for uid in user_ids:
            await propose_skills_from_facts(uid)

    except Exception as e:
        logger.error(f"[Skills] propose_skills_for_all_users failed: {e}", exc_info=True)
