"""Shared helpers for the proactive jobs (briefing, initiative loop, linker).

Each proactive job has a kill-switch env (default ON) so a misbehaving job can
be silenced without a redeploy, and all of them honor `PROACTIVE_ALLOWLIST`
(comma-separated user_ids; empty = all users) so the autopilot can be scoped to
a subset of accounts during testing without touching code.
"""
import os


def _truthy(value: str) -> bool:
    return str(value).strip().lower() not in ("0", "false", "no", "off", "")


def flag_enabled(env_name: str, default: bool = True) -> bool:
    raw = os.getenv(env_name)
    if raw is None:
        return default
    return _truthy(raw)


def proactive_allowlist() -> set[str]:
    raw = os.getenv("PROACTIVE_ALLOWLIST", "")
    return {x.strip() for x in raw.split(",") if x.strip()}


def is_allowed(user_id: str) -> bool:
    allow = proactive_allowlist()
    return (not allow) or (user_id in allow)
