"""Autonomy learner — daily job that mines action_approval_log for patterns
and writes proposed user_autonomy_rules.

Logic:
  For each (user_id, action_type, match_key) combination that has appeared
  in action_approval_log at least MIN_CONFIRMS times with no discards in the
  last LOOKBACK_DAYS days, upsert a row in user_autonomy_rules with
  mode='proposed'. Once the user enables it in the app (mode='auto'),
  _autonomy_allows_auto() in praxa_core bypasses the confirmation gate.
"""
import asyncio
import logging
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

MIN_CONFIRMS = int(__import__("os").getenv("AUTONOMY_MIN_CONFIRMS", "3"))
LOOKBACK_DAYS = int(__import__("os").getenv("AUTONOMY_LOOKBACK_DAYS", "30"))


async def learn_autonomy_patterns(user_id: str) -> int:
    """Scan action_approval_log for user and propose new autonomy rules.

    Returns the number of new rules proposed.
    """
    from .supabase_client import get_supabase_client
    db = get_supabase_client()

    since = (datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)).isoformat()

    try:
        resp = await asyncio.to_thread(
            lambda: db.client.table("action_approval_log")
            .select("action_type, match_key, outcome")
            .eq("user_id", user_id)
            .gte("created_at", since)
            .execute()
        )
        rows = resp.data or []
    except Exception as e:
        logger.error(f"[autonomy_learner] Failed to fetch log for user={user_id}: {e}")
        return 0

    from collections import defaultdict
    confirms: dict[tuple, int] = defaultdict(int)
    discards: dict[tuple, int] = defaultdict(int)

    for row in rows:
        key = (row.get("action_type", ""), row.get("match_key"))
        if row.get("outcome") == "confirmed":
            confirms[key] += 1
        elif row.get("outcome") == "discarded":
            discards[key] += 1

    proposed = 0
    for key, count in confirms.items():
        if count < MIN_CONFIRMS:
            continue
        if discards.get(key, 0) > 0:
            continue

        action_type, match_key = key
        if not action_type:
            continue

        label = _describe_rule(action_type, match_key)
        rule_row = {
            "user_id": user_id,
            "action_type": action_type,
            "match_key": match_key,
            "mode": "proposed",
            "label": label,
        }
        try:
            existing_resp = await asyncio.to_thread(
                lambda: db.client.table("user_autonomy_rules")
                .select("id")
                .eq("user_id", user_id)
                .eq("action_type", action_type)
                .eq("match_key", match_key)
                .maybe_single()
                .execute()
            )
            is_new = not bool(existing_resp and existing_resp.data)
            result = await asyncio.to_thread(
                lambda r=rule_row: db.client.table("user_autonomy_rules")
                .upsert(r, on_conflict="user_id,action_type,match_key")
                .execute()
            )
            rule_id = result.data[0].get("id") if result.data else None
            logger.info(
                f"[autonomy_learner] Proposed rule for user={user_id}: "
                f"{action_type} / match_key={match_key!r} ({count} confirms)"
            )
            proposed += 1

            if is_new and rule_id:
                await _notify_proposed_rule(user_id, label, rule_id)
        except Exception as e:
            logger.error(f"[autonomy_learner] Rule upsert failed for user={user_id}: {e}")

    return proposed


async def _notify_proposed_rule(user_id: str, label: str, rule_id: str) -> None:
    """Push a notification to the user so they can enable the proposed rule inline."""
    try:
        from services.push_service import get_user_push_token, send_push_notification
        push_token = await get_user_push_token(user_id)
        if not push_token:
            return
        await send_push_notification(
            push_token=push_token,
            title="Praxa learned your preference",
            body=f"{label} — tap to enable automatic execution.",
            data={
                "notificationType": "autonomy_rule_proposed",
                "ruleId": rule_id,
            },
        )
        logger.info(f"[autonomy_learner] Pushed rule proposal notification to user={user_id}")
    except Exception as e:
        logger.warning(f"[autonomy_learner] Push notification failed for user={user_id}: {e}")


def _describe_rule(action_type: str, match_key: str | None) -> str:
    if action_type == "send_slack_message":
        return f"Send Slack messages to #{match_key}" if match_key else "Send Slack messages"
    if action_type in ("reply_to_email", "send_email"):
        return f"Send emails to @{match_key} addresses" if match_key else "Send emails"
    if action_type == "reschedule_calendar_event":
        return "Reschedule calendar events"
    if action_type == "create_notion_page":
        return "Create Notion pages"
    if action_type == "update_notion_page":
        return "Update Notion pages"
    return f"Auto-execute {action_type.replace('_', ' ')}"


async def learn_autonomy_patterns_all_users() -> None:
    """Run pattern learning for every active user. Called by the scheduler."""
    from .supabase_client import get_supabase_client
    db = get_supabase_client()
    try:
        resp = await asyncio.to_thread(
            lambda: db.client.table("users").select("id").eq("ai_enabled", True).execute()
        )
        user_ids = [r["id"] for r in (resp.data or [])]
    except Exception as e:
        logger.error(f"[autonomy_learner] Could not load user list: {e}")
        return

    total = 0
    for uid in user_ids:
        try:
            n = await learn_autonomy_patterns(uid)
            total += n
        except Exception as e:
            logger.error(f"[autonomy_learner] Failed for user={uid}: {e}")

    if total:
        logger.info(f"[autonomy_learner] Proposed {total} new rules across {len(user_ids)} users")
