"""SMS agent - a tool-using, memory-loaded agent that treats SMS as a two-way
notification channel.

Replaces the one-shot JSON classifier (reply_processor) with an OpenAI
function-calling loop over the shared praxa_core tools. Because a tool-using
agent exceeds Twilio's synchronous webhook window, the inbound webhook acks
immediately and this runs in a BackgroundTask; the reply is sent via Twilio
outbound REST (twilio_sms.send_sms).

Behavior:
  - Reads (tasks, calendar, email) run direct.
  - Risky/external writes (email reply, Notion/Slack) queue as
    `pending_confirmation` via praxa_core; the agent tells the user and offers a
    deep link rather than editing complex content over text.
  - The user can reply "what is it?" to get the full detail of the latest
    pending action, or an affirmative ("yes", "send it") to confirm it.
"""
import asyncio
import logging
import os
from datetime import datetime, timezone

from .supabase_client import get_supabase_client

logger = logging.getLogger(__name__)

SMS_AGENT_MODEL = os.getenv("SMS_AGENT_MODEL", "gpt-4o-mini")
_MAX_ROUNDS = 5
_APP_LINK = os.getenv("PRAXA_APP_LINK", "the Praxa app")


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
        surface="sms",
        nylas_api_key=os.getenv("NYLAS_API_KEY"),
    )


async def _pending_actions(user_id: str) -> list[dict]:
    db = get_supabase_client()
    try:
        resp = await asyncio.to_thread(
            lambda: db.client.table("integration_actions")
            .select("id, action_type, provider, summary, payload, created_at")
            .eq("user_id", user_id)
            .eq("status", "pending_confirmation")
            .order("created_at", desc=True)
            .limit(5)
            .execute()
        )
        return resp.data or []
    except Exception as e:
        logger.error(f"[sms_agent] pending fetch failed: {e}")
        return []


# ---------------------------------------------------------------------------
# SMS-specific tools (the shared core tools live in praxa_core.agent_runtime)
# ---------------------------------------------------------------------------

_SMS_EXTRA_TOOLS = [
    {"type": "function", "function": {
        "name": "explain_pending_action",
        "description": "Get the full detail of a pending action awaiting the user's approval (use when they ask 'what is it?').",
        "parameters": {"type": "object", "properties": {"action_id": {"type": "string", "description": "Optional; defaults to the most recent pending action."}}},
    }},
    {"type": "function", "function": {
        "name": "confirm_pending_action",
        "description": "Confirm/approve a pending action so it executes. Use only when the user clearly approves.",
        "parameters": {"type": "object", "properties": {"action_id": {"type": "string", "description": "Optional; defaults to the most recent pending action."}}},
    }},
]


def _resolve_pending(action_id: str | None, pending: list[dict]) -> dict | None:
    if not pending:
        return None
    if action_id:
        return next((p for p in pending if p["id"] == action_id), pending[0])
    return pending[0]


def _explain_action(action_id: str | None, pending: list[dict]) -> str:
    action = _resolve_pending(action_id, pending)
    if not action:
        return "There's nothing waiting for your approval right now."
    payload = action.get("payload") or {}
    atype = action.get("action_type", "")
    if atype == "reply_to_email":
        return (
            f"It's a draft email reply to {payload.get('recipient', 'a contact')} "
            f"re: {payload.get('subject', '(no subject)')}. Draft: \"{(payload.get('reply_body') or '')[:300]}\". "
            f"Reply 'yes' to send, or open {_APP_LINK} to edit."
        )
    if atype == "create_notion_page":
        return (
            f"It's a new Notion page titled '{payload.get('title', '')}' with: "
            f"\"{(payload.get('content') or '')[:300]}\". Reply 'yes' to create it, or edit it in {_APP_LINK}."
        )
    if atype == "send_slack_message":
        return (
            f"It's a Slack message to {payload.get('channel', '')}: "
            f"\"{(payload.get('message') or '')[:300]}\". Reply 'yes' to send."
        )
    return f"{action.get('summary') or atype}. Reply 'yes' to approve, or open {_APP_LINK} to review."


async def _confirm_action(ctx, action_id: str | None, pending: list[dict]) -> str:
    action = _resolve_pending(action_id, pending)
    if not action:
        return "There's nothing waiting for your approval right now."
    db = get_supabase_client()
    now_iso = datetime.now(timezone.utc).isoformat()
    try:
        await asyncio.to_thread(
            lambda: db.client.table("integration_actions")
            .update({"status": "confirmed", "confirmed_at": now_iso, "updated_at": now_iso})
            .eq("id", action["id"])
            .execute()
        )
        from .action_dispatcher import dispatch_action_by_id
        await dispatch_action_by_id(action["id"])
        return f"Approved — {action.get('summary') or 'the action'} is on its way."
    except Exception as e:
        logger.error(f"[sms_agent] confirm failed: {e}")
        return "I couldn't confirm that just now. Please try again or use the app."


def _build_system_prompt(memory_context: str, pending: list[dict]) -> str:
    pending_block = ""
    if pending:
        lines = [f"- [{p['id']}] {p.get('summary') or p.get('action_type')}" for p in pending]
        pending_block = (
            "\n\nACTIONS AWAITING THE USER'S APPROVAL (most recent first):\n"
            + "\n".join(lines)
            + "\nIf the user asks 'what is it?' or for detail, call explain_pending_action. "
            "If they clearly approve ('yes', 'send it', 'go ahead'), call confirm_pending_action."
        )
    return (
        "You are Praxa, a warm, efficient assistant operating over SMS. Keep replies concise and "
        "text-friendly (ideally under ~300 chars). The user texts you to get things done.\n"
        "- Reads (tasks, calendar, email) are safe; call the tools to answer accurately, never guess.\n"
        "- Creating/completing tasks is safe to do directly.\n"
        "- Sending email, Notion, or Slack writes are queued for the user's approval automatically by the tools; "
        "tell the user it's drafted and they can reply 'yes' to approve or open the app to edit.\n"
        "- Never tell the user to 'open the app' as the only option for simple things they can do by replying.\n"
        "- Be specific and descriptive — if you took an action, say exactly what happened.\n"
        f"{memory_context}{pending_block}"
    )


async def handle_message(user_id: str, message: str) -> str:
    """Run the SMS agent for one inbound message and return the reply text.

    Delegates the conversation loop to the shared praxa_core agent runtime,
    layering SMS-only pending-action tools on top of the core tool set.
    """
    from .memory_service import load_session_context
    from praxa_core.agent_runtime import CORE_TOOLS, execute_core_tool, run_agent

    ctx = await _build_ctx(user_id)
    pending = await _pending_actions(user_id)

    try:
        memory_context = await load_session_context(user_id, query_text=message)
    except Exception as e:
        logger.warning(f"[sms_agent] memory load failed: {e}")
        memory_context = ""

    # Unified retrieval adds semantic hits from connected tools (Notion/Slack).
    try:
        from praxa_core.memory import get_integration_context
        tool_context = await get_integration_context(ctx, message)
        if tool_context:
            memory_context = f"{memory_context}\n\n{tool_context}".strip()
    except Exception as e:
        logger.warning(f"[sms_agent] integration context load failed: {e}")

    async def _execute(c, name: str, args: dict) -> str:
        if name == "explain_pending_action":
            return _explain_action(args.get("action_id"), pending)
        if name == "confirm_pending_action":
            return await _confirm_action(c, args.get("action_id"), pending)
        result = await execute_core_tool(c, name, args)
        return result if result is not None else f"Unknown tool: {name}"

    async def _refresh_pending() -> None:
        nonlocal pending
        pending = await _pending_actions(user_id)

    return await run_agent(
        ctx,
        message,
        system_prompt=_build_system_prompt(memory_context, pending),
        tools=CORE_TOOLS + _SMS_EXTRA_TOOLS,
        execute_tool=_execute,
        model=SMS_AGENT_MODEL,
        max_rounds=_MAX_ROUNDS,
        max_tokens=400,
        on_round_end=_refresh_pending,
    )


async def handle_inbound(user_id: str, body: str, from_number: str, triggered_by_dedupe_key: str | None = None) -> None:
    """Entry point for the inbound SMS BackgroundTask: run the agent, send the
    reply via Twilio, and log it. Falls back to the legacy reply_processor on error."""
    from .twilio_sms import send_sms

    reply: str
    intent = "agent"
    error = None
    try:
        reply = await handle_message(user_id, body)
    except Exception as e:
        logger.error(f"[sms_agent] agent failed, falling back to reply_processor: {e}", exc_info=True)
        error = str(e)
        try:
            from .reply_processor import process_reply
            reply = await process_reply(user_id=user_id, channel="sms", raw_message=body,
                                        triggered_by_dedupe_key=triggered_by_dedupe_key)
            intent = "fallback"
        except Exception as e2:
            logger.error(f"[sms_agent] fallback also failed: {e2}")
            reply = "Sorry, I hit an error handling that. Please try again or use the app."

    await send_sms(from_number, reply)
    await _log_reply(user_id, body, reply, intent, error)


async def _log_reply(user_id: str, raw_message: str, reply: str, intent: str, error: str | None) -> None:
    db = get_supabase_client()
    try:
        await asyncio.to_thread(
            lambda: db.client.table("notification_replies").insert({
                "user_id": user_id,
                "channel": "sms",
                "raw_message": raw_message,
                "intent": intent,
                "action_taken": "sms_agent",
                "action_result": {},
                "reply_sent": reply,
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "error": error,
            }).execute()
        )
    except Exception as e:
        logger.error(f"[sms_agent] log failed: {e}")
