"""
Unit tests for the Guardrail Validator.

Verifies that fabricated refund claims, ungrounded statements, and
prohibited content are correctly caught and blocked before delivery.

LLM judge calls are mocked. Deterministic layers (layer 1 & 3) run for real.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from agents.guardrails import (
    GroundingVerdict,
    _check_prohibited_content,
    _check_refund_guardrail,
    validate_guardrails,
)
from agents.state import EmailState

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

REAL_REFUND_CHUNK = {
    "content": (
        "Customers are eligible for a full refund if the refund request is submitted "
        "within 30 days of the original purchase date. "
        "Credit/debit card refunds take 5–10 business days after approval."
    ),
    "metadata": {"source_file": "refund_policy.md", "is_refund_policy": True},
    "score": 0.92,
}

GENERIC_CHUNK = {
    "content": "Our app is available on iOS and Android devices.",
    "metadata": {"source_file": "sample_faq.md", "is_refund_policy": False},
    "score": 0.85,
}


def _make_state(
    draft: str,
    chunks: list[dict] | None = None,
    email_id: str = "guard-001",
) -> EmailState:
    return {
        "email_id": email_id,
        "sender_email": "customer@example.com",
        "subject": "Refund request",
        "body": "Can I get a refund?",
        "received_at": "2026-08-05T12:00:00Z",
        "draft_response": draft,
        "retrieved_chunks": chunks if chunks is not None else [REAL_REFUND_CHUNK],
        "citations": ["refund_policy.md"],
        "audit_log": [],
    }


# ---------------------------------------------------------------------------
# Layer 1: Refund Policy Guardrail (deterministic)
# ---------------------------------------------------------------------------

class TestRefundGuardrail:

    def test_grounded_refund_claim_passes(self):
        draft = "You can get a full refund within 30 days of your purchase."
        violations = _check_refund_guardrail(draft, [REAL_REFUND_CHUNK])
        # "30 days" is in the refund chunk text — no violations
        assert violations == []

    def test_fabricated_day_count_flagged(self):
        """A timeline not in any refund chunk should be flagged."""
        draft = "You are eligible for a refund within 90 days of purchase."
        violations = _check_refund_guardrail(draft, [REAL_REFUND_CHUNK])
        # "90 days" is NOT in the chunk — should be flagged
        assert len(violations) > 0
        assert any("90" in v or "days" in v.lower() for v in violations)

    def test_no_refund_keywords_skips_check(self):
        """Draft without refund terms should skip the refund check entirely."""
        draft = "Your account has been set up successfully. Welcome aboard!"
        violations = _check_refund_guardrail(draft, [])
        assert violations == []

    def test_missing_policy_chunks_flagged(self):
        """Refund claim with no refund-policy chunks retrieved → violation."""
        draft = "You are eligible for a refund."
        # Chunk present, but not tagged as is_refund_policy
        violations = _check_refund_guardrail(draft, [GENERIC_CHUNK])
        assert len(violations) > 0

    def test_no_chunks_at_all_with_refund_term(self):
        """Refund keyword + zero chunks → cannot verify → violation."""
        draft = "You can request a refund via email."
        violations = _check_refund_guardrail(draft, [])
        assert len(violations) > 0


# ---------------------------------------------------------------------------
# Layer 3: Prohibited Content Scan (deterministic)
# ---------------------------------------------------------------------------

class TestProhibitedContentScan:

    def test_dollar_amount_not_in_context_flagged(self):
        """Dollar amount that doesn't appear in retrieved context is flagged."""
        draft = "We will refund $500 to your account within 3 days."
        violations = _check_prohibited_content(draft, [GENERIC_CHUNK])
        assert any("$500" in v for v in violations)

    def test_dollar_amount_in_context_passes(self):
        """Dollar amount that appears verbatim in retrieved context is NOT flagged."""
        chunk = {
            "content": "The Professional plan costs $79/month.",
            "metadata": {"is_refund_policy": False},
            "score": 0.9,
        }
        draft = "The Professional plan is $79/month."
        violations = _check_prohibited_content(draft, [chunk])
        assert all("$79" not in v for v in violations)

    def test_legal_claim_flagged(self):
        """Unsupported legal claims are always flagged."""
        draft = "You are legally obligated to receive a full refund within 24 hours."
        violations = _check_prohibited_content(draft, [REAL_REFUND_CHUNK])
        assert any("legal" in v.lower() for v in violations)

    def test_fabricated_phone_number_flagged(self):
        """Phone numbers not in context are flagged."""
        draft = "Please call us at 1-800-555-0199 for immediate assistance."
        violations = _check_prohibited_content(draft, [GENERIC_CHUNK])
        assert len(violations) > 0


# ---------------------------------------------------------------------------
# Full validate_guardrails node
# ---------------------------------------------------------------------------

class TestValidateGuardrails:

    @patch("agents.guardrails._judge_chain")
    def test_clean_draft_passes(self, mock_judge):
        """A fully grounded draft with no violations should pass."""
        mock_judge.invoke.return_value = GroundingVerdict(
            is_grounded=True,
            ungrounded_statements=[],
            verdict_reasoning="All claims are grounded.",
        )
        state = _make_state(
            draft="Thank you for contacting us. Based on our policy, refunds are processed within 30 days. Best regards, Customer Support Team",
            chunks=[REAL_REFUND_CHUNK],
        )
        result = validate_guardrails(state)
        assert result["guardrail_passed"] is True
        assert result["guardrail_violations"] == []

    @patch("agents.guardrails._judge_chain")
    def test_llm_judge_failure_blocks_response(self, mock_judge):
        """If the LLM judge finds ungrounded statements, guardrail fails."""
        mock_judge.invoke.return_value = GroundingVerdict(
            is_grounded=False,
            ungrounded_statements=["Claim that processing takes 1 hour is not in context."],
            verdict_reasoning="Fabricated timeline found.",
        )
        state = _make_state(
            draft="Your refund will be processed within 1 hour.",
            chunks=[REAL_REFUND_CHUNK],
        )
        result = validate_guardrails(state)
        assert result["guardrail_passed"] is False
        assert result["final_action"] == "route_to_human"
        assert len(result["guardrail_violations"]) > 0

    @patch("agents.guardrails._judge_chain")
    def test_guardrail_failure_sets_human_route(self, mock_judge):
        """Failed guardrail should always set final_action to route_to_human."""
        mock_judge.invoke.return_value = GroundingVerdict(
            is_grounded=False,
            ungrounded_statements=["Invented policy detail."],
            verdict_reasoning="Hallucination detected.",
        )
        state = _make_state(
            draft="You are legally obligated to get a $999 refund immediately.",
            chunks=[GENERIC_CHUNK],
        )
        result = validate_guardrails(state)
        assert result["final_action"] == "route_to_human"

    @patch("agents.guardrails._judge_chain")
    def test_audit_log_appended(self, mock_judge):
        """Audit log should have the guardrail entry appended."""
        mock_judge.invoke.return_value = GroundingVerdict(
            is_grounded=True,
            ungrounded_statements=[],
            verdict_reasoning="All grounded.",
        )
        state = _make_state(
            draft="Thank you for contacting support. Best regards, Customer Support Team",
            chunks=[GENERIC_CHUNK],
        )
        state["audit_log"] = ["previous-entry"]
        result = validate_guardrails(state)
        assert len(result["audit_log"]) == 2
        assert "validate_guardrails" in result["audit_log"][-1]
