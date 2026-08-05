"""
Unit tests for the Escalation Checker (deterministic rule engine).

All escalation logic is deterministic — no mocking of LLM needed.
Redis is replaced by the in-memory fallback via mocking get_contact_count.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch

from agents.escalation_checker import check_escalation
from agents.state import EmailState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(
    subject: str = "",
    body: str = "",
    sender: str = "customer@example.com",
    email_id: str = "esc-001",
    contact_count: int = 0,
) -> EmailState:
    return {
        "email_id": email_id,
        "sender_email": sender,
        "subject": subject,
        "body": body,
        "received_at": "2026-08-05T12:00:00Z",
        "category": "technical",
        "audit_log": [],
    }


# Patch contact tracker to control the count without Redis
_PATCH_TARGET = "agents.escalation_checker.get_contact_count"


# ---------------------------------------------------------------------------
# Rule 1: Data Loss
# ---------------------------------------------------------------------------

class TestDataLossRule:

    @pytest.mark.parametrize("text", [
        "I experienced data loss yesterday",
        "My data is missing from the dashboard",
        "All my data was deleted",
        "My data is gone after the update",
        "I lost my files after the upgrade",
    ])
    @patch(_PATCH_TARGET, return_value=0)
    def test_data_loss_triggers_escalation(self, _, text):
        state = _make_state(body=text)
        result = check_escalation(state)
        assert result["is_critical"] is True
        assert any("data loss" in r.lower() for r in result["escalation_reasons"])

    @patch(_PATCH_TARGET, return_value=0)
    def test_no_false_positive_for_data_terms(self, _):
        state = _make_state(body="I have a lot of data to analyze, everything is fine.")
        result = check_escalation(state)
        assert result["is_critical"] is False


# ---------------------------------------------------------------------------
# Rule 2: Service Outage
# ---------------------------------------------------------------------------

class TestServiceOutageRule:

    @pytest.mark.parametrize("text", [
        "There's an outage affecting our team",
        "The service is down right now",
        "The system is down and we can't work",
        "We can't access the platform",
        "Experiencing significant downtime",
        "The site is completely unavailable",
    ])
    @patch(_PATCH_TARGET, return_value=0)
    def test_outage_triggers_escalation(self, _, text):
        state = _make_state(body=text)
        result = check_escalation(state)
        assert result["is_critical"] is True
        assert any("outage" in r.lower() for r in result["escalation_reasons"])

    @patch(_PATCH_TARGET, return_value=0)
    def test_slow_performance_not_outage(self, _):
        state = _make_state(body="The app is a bit slow today, minor inconvenience.")
        result = check_escalation(state)
        assert result["is_critical"] is False


# ---------------------------------------------------------------------------
# Rule 3: Security Breach
# ---------------------------------------------------------------------------

class TestSecurityBreachRule:

    @pytest.mark.parametrize("text", [
        "I think there's been a security breach",
        "My account was hacked",
        "There's unauthorized access to my account",
        "I think my account was compromised",
        "There seems to be a data leak",
        "I received a phishing email from your domain",
    ])
    @patch(_PATCH_TARGET, return_value=0)
    def test_security_breach_triggers_escalation(self, _, text):
        state = _make_state(body=text)
        result = check_escalation(state)
        assert result["is_critical"] is True
        assert any("security" in r.lower() for r in result["escalation_reasons"])

    @patch(_PATCH_TARGET, return_value=0)
    def test_security_tips_request_not_breach(self, _):
        state = _make_state(body="Can you share some security tips for our team?")
        result = check_escalation(state)
        assert result["is_critical"] is False


# ---------------------------------------------------------------------------
# Rule 4: Contact Frequency
# ---------------------------------------------------------------------------

class TestContactFrequencyRule:

    @patch(_PATCH_TARGET, return_value=4)  # 4 > threshold of 3
    def test_high_frequency_triggers_escalation(self, _):
        state = _make_state(body="Just a regular billing question.")
        result = check_escalation(state)
        assert result["is_critical"] is True
        assert any("frequency" in r.lower() or "contact" in r.lower() for r in result["escalation_reasons"])

    @patch(_PATCH_TARGET, return_value=3)  # exactly at threshold — NOT triggered
    def test_at_threshold_not_triggered(self, _):
        state = _make_state(body="Just a regular billing question.")
        result = check_escalation(state)
        assert result["is_critical"] is False

    @patch(_PATCH_TARGET, return_value=1)
    def test_low_frequency_not_triggered(self, _):
        state = _make_state(body="Refund request.")
        result = check_escalation(state)
        assert result["is_critical"] is False


# ---------------------------------------------------------------------------
# Multiple rules
# ---------------------------------------------------------------------------

class TestMultipleRules:

    @patch(_PATCH_TARGET, return_value=5)
    def test_multiple_rules_all_reasons_captured(self, _):
        state = _make_state(
            body="The service is down and I think my account was hacked!"
        )
        result = check_escalation(state)
        assert result["is_critical"] is True
        # Should capture outage AND security breach reasons
        assert len(result["escalation_reasons"]) >= 2

    @patch(_PATCH_TARGET, return_value=0)
    def test_safe_email_passes(self, _):
        state = _make_state(
            subject="Feature request",
            body="Would love to have a dark mode option.",
        )
        result = check_escalation(state)
        assert result["is_critical"] is False
        assert result["final_action"] != "route_to_human" if "final_action" in result else True


# ---------------------------------------------------------------------------
# Queue assignment
# ---------------------------------------------------------------------------

class TestQueueAssignment:

    @patch(_PATCH_TARGET, return_value=0)
    def test_security_breach_routes_to_security_team(self, _):
        state = _make_state(body="My account was hacked and compromised!")
        result = check_escalation(state)
        assert result["assigned_agent_queue"] == "security-team"

    @patch(_PATCH_TARGET, return_value=0)
    def test_data_loss_routes_to_technical_tier2(self, _):
        state = _make_state(body="All my data is gone after your update.")
        result = check_escalation(state)
        assert result["assigned_agent_queue"] == "technical-tier-2"

    @patch(_PATCH_TARGET, return_value=5)
    def test_frequency_only_routes_to_billing_tier2(self, _):
        state = _make_state(body="What is my current plan?")
        result = check_escalation(state)
        assert result["assigned_agent_queue"] == "billing-tier-2"
