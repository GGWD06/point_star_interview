"""
Escalation Checker — deterministic rule engine.

IMPORTANT: No LLM is involved in escalation decisions.
All rules use keyword/regex matching and Redis lookups for
predictable, auditable, safety-critical routing.

Rules:
  1. Email mentions data loss
  2. Email mentions service outage
  3. Email mentions security breach
  4. Customer contacted >3 times in the last 7 days
"""

from __future__ import annotations

import logging
import re

from agents.state import EmailState
from config.settings import settings
from services.contact_tracker import get_contact_count

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyword/regex rule sets
# ---------------------------------------------------------------------------

_RULE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "data_loss",
        re.compile(
            r"\b(data\s+loss|lost\s+data|data\s+(is\s+|was\s+)?missing|data\s+(is\s+|was\s+)?deleted|"
            r"data\s+(is\s+|was\s+)?gone|my\s+files\s+(are\s+|were\s+)?gone|lost\s+my\s+files)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "service_outage",
        re.compile(
            r"\b(outage|service\s+(is\s+|was\s+)?down|system\s+(is\s+|was\s+)?down|can'?t\s+access|"
            r"not\s+working|downtime|site\s+is\s+down|completely\s+down|"
            r"unavailable|can\s+not\s+access)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "security_breach",
        re.compile(
            r"\b(security\s+breach|hacked|unauthorized\s+access|compromised|"
            r"data\s+leak|account\s+breach|someone\s+else\s+(is\s+|was\s+)?logged\s+in|"
            r"stolen\s+credentials|phishing)\b",
            re.IGNORECASE,
        ),
    ),
]

# Human-readable labels for audit log
_RULE_LABELS = {
    "data_loss": "Possible data loss event reported by customer.",
    "service_outage": "Possible service outage reported by customer.",
    "security_breach": "Possible security breach reported by customer.",
    "frequency": (
        f"Customer exceeded contact frequency threshold "
        f"(>{settings.escalation_contact_threshold} contacts in "
        f"{settings.escalation_contact_window_days} days)."
    ),
}

# Queue assignments by rule
_RULE_QUEUES = {
    "data_loss": "technical-tier-2",
    "service_outage": "operations-team",
    "security_breach": "security-team",
    "frequency": "billing-tier-2",  # default; may be overridden by category
    "default": "human_review",
}


def _pick_queue(triggered_rules: list[str], category: str) -> str:
    """
    Select the most appropriate human queue.
    Security and data issues take priority over generic frequency escalation.
    """
    priority_order = ["security_breach", "data_loss", "service_outage", "frequency"]
    for rule in priority_order:
        if rule in triggered_rules:
            return _RULE_QUEUES[rule]
    return _RULE_QUEUES["default"]


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------

def check_escalation(state: EmailState) -> EmailState:
    """
    LangGraph node: applies all escalation rules deterministically.

    Reads:  sender_email, subject, body, category, audit_log
    Writes: is_critical, escalation_reasons, contact_count_7d,
            final_action (conditionally), assigned_agent_queue, audit_log
    """
    email_id = state["email_id"]
    sender = state["sender_email"]
    search_text = f"{state.get('subject', '')} {state.get('body', '')}"

    triggered: list[str] = []
    reasons: list[str] = []

    # --- Rule 1–3: keyword/regex matching ---
    for rule_name, pattern in _RULE_PATTERNS:
        if pattern.search(search_text):
            triggered.append(rule_name)
            reasons.append(_RULE_LABELS[rule_name])
            logger.info("[%s] Escalation rule triggered: %s", email_id, rule_name)

    # --- Rule 4: contact frequency ---
    contact_count = get_contact_count(
        email=sender,
        window_days=settings.escalation_contact_window_days,
    )
    if contact_count > settings.escalation_contact_threshold:
        triggered.append("frequency")
        reasons.append(_RULE_LABELS["frequency"])
        logger.info(
            "[%s] Escalation rule triggered: frequency (count=%d)", email_id, contact_count
        )

    is_critical = len(triggered) > 0
    updates: dict = {
        "is_critical": is_critical,
        "escalation_reasons": reasons,
        "contact_count_7d": contact_count,
    }

    log_entry = (
        f"check_escalation: is_critical={is_critical} "
        f"rules={triggered} contact_count_7d={contact_count}"
    )

    if is_critical:
        queue = _pick_queue(triggered, state.get("category", "general"))
        updates["final_action"] = "route_to_human"
        updates["assigned_agent_queue"] = queue
        logger.warning("[%s] Email escalated → queue: %s", email_id, queue)

    return {
        **state,
        **updates,
        "audit_log": [*state.get("audit_log", []), log_entry],
    }
