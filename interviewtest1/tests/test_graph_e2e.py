"""
End-to-end pipeline tests.

Tests the full LangGraph graph from email intake to final routing decision.
All external calls (LLM, Redis, ChromaDB) are mocked.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from agents.graph import process_email
from agents.state import EmailState
from agents.classifier import ClassificationResult
from agents.guardrails import GroundingVerdict

# ---------------------------------------------------------------------------
# Shared mock data
# ---------------------------------------------------------------------------

MOCK_CLASSIFICATION_BILLING = ClassificationResult(
    category="billing",
    confidence=0.97,
    detected_keywords=["refund", "charged"],
    reasoning="Email is about a billing issue.",
)

MOCK_CLASSIFICATION_GENERAL = ClassificationResult(
    category="general",
    confidence=0.85,
    detected_keywords=["how to", "invite"],
    reasoning="General onboarding question.",
)

MOCK_CHUNKS = [
    {
        "content": "Refunds are processed within 30 days of purchase. Credit card refunds take 5–10 business days.",
        "metadata": {"source_file": "refund_policy.md", "is_refund_policy": True},
        "score": 0.92,
    }
]

MOCK_GROUNDING_PASS = GroundingVerdict(
    is_grounded=True,
    ungrounded_statements=[],
    verdict_reasoning="All claims grounded.",
)

MOCK_GROUNDING_FAIL = GroundingVerdict(
    is_grounded=False,
    ungrounded_statements=["Claim about 1-hour processing not in context."],
    verdict_reasoning="Hallucination detected.",
)


def _base_email(
    subject="How do I invite a teammate?",
    body="I want to add someone to my team account.",
    email_id="e2e-001",
    sender="user@example.com",
) -> EmailState:
    return {
        "email_id": email_id,
        "sender_email": sender,
        "subject": subject,
        "body": body,
        "received_at": "2026-08-05T12:00:00Z",
        "audit_log": [],
    }


# ---------------------------------------------------------------------------
# Patch targets
# ---------------------------------------------------------------------------

PATCH_CLASSIFIER = "agents.classifier._chain"
PATCH_JUDGE = "agents.guardrails._judge_chain"
PATCH_CONTACT = "agents.escalation_checker.get_contact_count"
PATCH_RETRIEVER = "agents.retrieve_context.get_retriever"


def _mock_retriever(chunks=None):
    from rag.retriever import RetrievalResult
    ret = MagicMock()
    ret.retrieve.return_value = RetrievalResult(
        chunks=chunks or MOCK_CHUNKS,
        citations=["refund_policy.md"],
        num_retrieved=len(chunks or MOCK_CHUNKS),
    )
    return ret


# ---------------------------------------------------------------------------
# E2E Test: Happy path — auto-reply
# ---------------------------------------------------------------------------

class TestHappyPathAutoReply:

    @patch(PATCH_CONTACT, return_value=0)
    @patch(PATCH_CLASSIFIER)
    @patch(PATCH_RETRIEVER)
    @patch(PATCH_JUDGE)
    @patch("agents.response_drafter._llm")
    def test_general_email_auto_replied(
        self, mock_llm, mock_judge_chain, mock_get_retriever, mock_classifier, mock_contact
    ):
        """A safe general email should be classified, retrieved, drafted, and auto-replied."""
        mock_classifier.invoke.return_value = MOCK_CLASSIFICATION_GENERAL
        mock_get_retriever.return_value = _mock_retriever([
            {
                "content": "You can invite a team member via Settings → Team → Invite.",
                "metadata": {"source_file": "sample_faq.md", "is_refund_policy": False},
                "score": 0.90,
            }
        ])
        mock_response = MagicMock()
        mock_response.content = "You can invite teammates via Settings → Team → Invite. Best regards, Customer Support Team"
        mock_llm.invoke.return_value = mock_response

        # Build chain mock for drafter
        with patch("agents.response_drafter.ChatPromptTemplate") as mock_prompt_cls:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = mock_response
            mock_prompt_cls.from_messages.return_value.__or__ = lambda self, other: mock_chain

            mock_judge_chain.invoke.return_value = MOCK_GROUNDING_PASS

            state = _base_email()
            result = process_email(state)

        assert result["category"] == "general"
        assert result["is_critical"] is False
        assert result["final_action"] == "send_auto_reply"
        assert len(result["audit_log"]) >= 4  # classify, escalate, retrieve, draft, guardrail, send, audit
        assert "audit_complete" in result["audit_log"][-1]


# ---------------------------------------------------------------------------
# E2E Test: Escalation — security breach
# ---------------------------------------------------------------------------

class TestEscalationPath:

    @patch(PATCH_CONTACT, return_value=0)
    @patch(PATCH_CLASSIFIER)
    def test_security_breach_routes_to_human(self, mock_classifier, mock_contact):
        """Security breach keywords should skip RAG and route directly to human."""
        mock_classifier.invoke.return_value = ClassificationResult(
            category="technical",
            confidence=0.96,
            detected_keywords=["hacked", "unauthorized access"],
            reasoning="Security breach reported.",
        )
        state = _base_email(
            subject="My account was hacked",
            body="Someone gained unauthorized access to my account. I think I was hacked.",
        )
        result = process_email(state)

        assert result["is_critical"] is True
        assert result["final_action"] == "route_to_human"
        assert result["assigned_agent_queue"] == "security-team"
        # Should NOT have reached response drafting
        assert result.get("draft_response") is None

    @patch(PATCH_CONTACT, return_value=5)  # High frequency
    @patch(PATCH_CLASSIFIER)
    def test_high_frequency_routes_to_human(self, mock_classifier, mock_contact):
        """Customer with >3 contacts in 7 days should be escalated regardless of content."""
        mock_classifier.invoke.return_value = MOCK_CLASSIFICATION_BILLING
        state = _base_email(body="Just a billing question about my invoice.")
        result = process_email(state)

        assert result["is_critical"] is True
        assert result["final_action"] == "route_to_human"
        assert result["contact_count_7d"] == 5


# ---------------------------------------------------------------------------
# E2E Test: Guardrail blocking
# ---------------------------------------------------------------------------

class TestGuardrailBlocking:

    @patch(PATCH_CONTACT, return_value=0)
    @patch(PATCH_CLASSIFIER)
    @patch(PATCH_RETRIEVER)
    @patch(PATCH_JUDGE)
    def test_hallucinated_response_blocked(
        self, mock_judge_chain, mock_get_retriever, mock_classifier, mock_contact
    ):
        """A hallucinated draft should be caught by guardrails and routed to human."""
        mock_classifier.invoke.return_value = MOCK_CLASSIFICATION_BILLING
        mock_get_retriever.return_value = _mock_retriever()
        mock_judge_chain.invoke.return_value = MOCK_GROUNDING_FAIL

        hallucinated_draft = (
            "You are legally obligated to receive a refund of $999 within 1 hour."
        )

        with patch("agents.response_drafter.ChatPromptTemplate") as mock_prompt_cls:
            mock_chain = MagicMock()
            mock_response = MagicMock()
            mock_response.content = hallucinated_draft
            mock_chain.invoke.return_value = mock_response
            mock_prompt_cls.from_messages.return_value.__or__ = lambda self, other: mock_chain

            state = _base_email(
                subject="I want a refund",
                body="Can I get my money back?",
            )
            result = process_email(state)

        assert result["guardrail_passed"] is False
        assert result["final_action"] == "route_to_human"
        assert len(result["guardrail_violations"]) > 0


# ---------------------------------------------------------------------------
# E2E Test: Audit trail completeness
# ---------------------------------------------------------------------------

class TestAuditTrail:

    @patch(PATCH_CONTACT, return_value=0)
    @patch(PATCH_CLASSIFIER)
    def test_escalated_email_audit_trail(self, mock_classifier, mock_contact):
        """Audit log should contain entries from each node that executed."""
        mock_classifier.invoke.return_value = MOCK_CLASSIFICATION_GENERAL
        state = _base_email(
            subject="Service outage",
            body="The service is completely down and we cannot access anything.",
        )
        result = process_email(state)

        audit = result["audit_log"]
        assert any("classify_email" in entry for entry in audit)
        assert any("check_escalation" in entry for entry in audit)
        assert any("route_to_human" in entry for entry in audit)
        assert any("audit_complete" in entry for entry in audit)
