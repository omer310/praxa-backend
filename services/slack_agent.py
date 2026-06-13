"""Slack conversational surface.

Lets a user talk to Praxa in Slack (DM or @-mention) with the same brain as
SMS: the shared praxa_core agent runtime over the core tool set. Reads run
directly; risky writes queue as pending_confirmation for in-app approval.

Inbound events arrive via the `slack-events` Supabase edge function, which
verifies the Slack signature and forwards the event here. We resolve the
workspace to a Praxa user via user_integrations, run the agent, and reply
in-thread using that workspace's bot token.

Kill-switch: SLACK_AGENT_ENABLED (default on). Requires the Slack app's Event
Subscriptions request URL to point at the edge function (external setup).
"""
import asyncio
import logging
import os
import re

import httpx

from .proactive import flag_enabled
from .supabase_client import get_supabase_client

logger = logging.getLogger(__name__)

SLACK_AGENT_MODEL = os.getenv("SLACK_AGENT_MODEL", "gpt-4o-mini")
_MENTION_RE = re.compile(r"<@[A-Z0-9]+>")


def enabled() -> bool:
    return flag_enabled("SLACK_AGENT_ENABLED", default=True)


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
        surface="slack",
        nylas_api_key=os.getenv("NYLAS_API_KEY"),
    )


async def _resolve_workspace(team_id: str | None) -> dict | None:
    if not team_id:
        return None
    db = get_supabase_client()
    try:
        resp = await asyncio.to_thread(
            lambda: db.client.table("user_integrations")
            .select("user_id, access_token, bot_user_id, workspace_id")
            .eq("provider", "slack")
            .eq("status", "connected")
            .eq("workspace_id", team_id)
            .limit(1)
            .execute()
        )
        rows = resp.data or []
        return rows[0] if rows else None
    except Exception as e:
        logger.error(f"[slack_agent] workspace resolve failed: {e}")
        return None


def _strip_mention(text: str) -> str:
    return _MENTION_RE.sub("", text or "").strip()


async def _post_message(token: str, channel: str, text: str, thread_ts: str | None) -> None:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                "https://slack.com/api/chat.postMessage",
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                json={"channel": channel, "text": text[:3500], "thread_ts": thread_ts},
            )
            data = resp.json()
            if not data.get("ok"):
                logger.error(f"[slack_agent] postMessage failed: {data.get('error')}")
    except Exception as e:
        logger.error(f"[slack_agent] postMessage error: {e}")


def _system_prompt(memory_context: str) -> str:
    return (
        "You are Praxa, a warm, efficient assistant operating inside Slack. Keep replies concise and "
        "Slack-friendly (you may use *bold* and short bullet lists). The user is talking to you to get things done.\n"
        "- Reads (tasks, calendar, email, Notion) are safe; call the tools to answer accurately, never guess.\n"
        "- Creating/completing tasks is safe to do directly.\n"
        "- Sending email or other external writes are queued for the user's approval automatically by the tools; "
        "tell the user it's drafted and they can approve it in the Praxa app.\n"
        "- Be specific about what you did.\n"
        f"{memory_context}"
    )


async def _run(user_id: str, text: str) -> str:
    from praxa_core.agent_runtime import CORE_TOOLS, run_agent
    from .memory_service import load_session_context

    ctx = await _build_ctx(user_id)
    try:
        memory_context = await load_session_context(user_id, query_text=text)
    except Exception as e:
        logger.warning(f"[slack_agent] memory load failed: {e}")
        memory_context = ""
    try:
        from praxa_core.memory import get_integration_context
        extra = await get_integration_context(ctx, text)
        if extra:
            memory_context = f"{memory_context}\n\n{extra}".strip()
    except Exception as e:
        logger.warning(f"[slack_agent] integration context failed: {e}")

    return await run_agent(
        ctx,
        text,
        system_prompt=_system_prompt(memory_context),
        tools=CORE_TOOLS,
        model=SLACK_AGENT_MODEL,
        max_tokens=600,
    )


async def handle_event(payload: dict) -> None:
    """Process one forwarded Slack event (runs in a BackgroundTask)."""
    if not enabled():
        return

    event = payload.get("event") or {}
    etype = event.get("type")

    # Ignore bot messages, edits, joins, and other non-user chatter.
    if event.get("bot_id") or event.get("subtype"):
        return
    if etype not in ("app_mention", "message"):
        return
    # For plain messages, only respond in DMs; channels require an @-mention.
    if etype == "message" and event.get("channel_type") != "im":
        return

    text = (event.get("text") or "").strip()
    channel = event.get("channel")
    if not text or not channel:
        return

    team_id = payload.get("team_id")
    if not team_id:
        auths = payload.get("authorizations") or []
        team_id = (auths[0] or {}).get("team_id") if auths else None

    ws = await _resolve_workspace(team_id)
    if not ws:
        logger.info(f"[slack_agent] no connected workspace for team {team_id}")
        return
    if event.get("user") and event.get("user") == ws.get("bot_user_id"):
        return

    clean = _strip_mention(text)
    if not clean:
        return

    try:
        reply = await _run(ws["user_id"], clean)
    except Exception as e:
        logger.error(f"[slack_agent] run failed: {e}", exc_info=True)
        reply = "Sorry, I hit an error handling that. Try again or open the Praxa app."

    thread_ts = event.get("thread_ts") or event.get("ts")
    await _post_message(ws["access_token"], channel, reply, thread_ts)
