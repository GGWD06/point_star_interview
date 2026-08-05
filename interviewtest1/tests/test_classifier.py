"""
Unit tests for the Classifier Agent.

Tests classification accuracy across all four categories using mocked
LLM responses (no real API calls required).
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from agents.classifier import ClassificationResult, classify_email
from agents.state import EmailState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(subject: str, body: str, email_id: str = "test-001") -> EmailState:
    return {
        "email_id": email_id,
        "sender_email": "customer@example.com",
        "subject": subject,
        "body": body,
        "received_at": "2026-08-05T12:00:00Z",
        "audit_log": [],
    }


def _mock_result(category, confidence=0.95, keywords=None, reasoning="test"):
    return ClassificationResult(
        category=category,
        confidence=confidence,
        detected_keywords=keywords or [category],
        reasoning=reasoning,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestClassifyEmail:

    @patch("agents.classifier._chain")
    def test_billing_classification(self, mock_chain):
        """Should classify payment/refund emails as 'billing'."""
        mock_chain.invoke.return_value = _mock_result(
            "billing", 0.97, ["charged twice", "refund"], "Mentions double charge and refund."
        )
        state = _make_state(
            subject="I was charged twice this month",
            body="Hello, I noticed two charges of $79. Please refund one of them.",
        )
        result = classify_email(state)

        assert result["category"] == "billing"
        assert result["confidence"] == 0.97
        assert "charged twice" in result["detected_keywords"]
        assert len(result["audit_log"]) == 1
        assert "billing" in result["audit_log"][0]

    @patch("agents.classifier._chain")
    def test_technical_classification(self, mock_chain):
        """Should classify bug/error emails as 'technical'."""
        mock_chain.invoke.return_value = _mock_result(
            "technical", 0.95, ["crashes", "error 500"]
        )
        state = _make_state(
            subject="App crashes on startup",
            body="Since the last update, your app crashes with error code 500.",
        )
        result = classify_email(state)

        assert result["category"] == "technical"
        assert result["confidence"] == 0.95

    @patch("agents.classifier._chain")
    def test_feedback_classification(self, mock_chain):
        """Should classify feature requests as 'feedback'."""
        mock_chain.invoke.return_value = _mock_result(
            "feedback", 0.92, ["dark mode", "feature request"]
        )
        state = _make_state(
            subject="Feature request: dark mode",
            body="I'd love a dark mode option for the app.",
        )
        result = classify_email(state)

        assert result["category"] == "feedback"

    @patch("agents.classifier._chain")
    def test_general_classification(self, mock_chain):
        """Should classify onboarding questions as 'general'."""
        mock_chain.invoke.return_value = _mock_result(
            "general", 0.88, ["how to", "invite"]
        )
        state = _make_state(
            subject="How do I add a team member?",
            body="I just signed up. How do I invite my colleague?",
        )
        result = classify_email(state)

        assert result["category"] == "general"

    @patch("agents.classifier._chain")
    def test_audit_log_is_appended(self, mock_chain):
        """Should append to existing audit_log, not replace it."""
        mock_chain.invoke.return_value = _mock_result("general")
        state = _make_state("Hello", "Just saying hi")
        state["audit_log"] = ["pre-existing-entry"]

        result = classify_email(state)

        assert len(result["audit_log"]) == 2
        assert result["audit_log"][0] == "pre-existing-entry"

    @patch("agents.classifier._chain")
    def test_state_fields_preserved(self, mock_chain):
        """Should preserve all existing state fields."""
        mock_chain.invoke.return_value = _mock_result("billing")
        state = _make_state("Invoice issue", "My invoice is wrong.")
        state["contact_count_7d"] = 2

        result = classify_email(state)

        assert result["email_id"] == "test-001"
        assert result["sender_email"] == "customer@example.com"
        assert result["contact_count_7d"] == 2

    @patch("agents.classifier._chain")
    def test_classification_reasoning_stored(self, mock_chain):
        """Should store the classification_reasoning field."""
        mock_chain.invoke.return_value = _mock_result(
            "technical", reasoning="Email describes a software crash."
        )
        state = _make_state("Crash report", "App is crashing.")
        result = classify_email(state)

        assert result["classification_reasoning"] == "Email describes a software crash."
