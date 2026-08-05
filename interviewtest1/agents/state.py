"""
Shared EmailState TypedDict — the single source of truth flowing
through the LangGraph StateGraph.
"""

from __future__ import annotations

from typing import Literal
from typing_extensions import TypedDict


class EmailState(TypedDict, total=False):
    # ------------------------------------------------------------------
    # Input — populated before the graph is invoked
    # ------------------------------------------------------------------
    email_id: str
    sender_email: str
    subject: str
    body: str
    received_at: str  # ISO-8601 timestamp string

    # ------------------------------------------------------------------
    # Classification — populated by the Classifier Agent
    # ------------------------------------------------------------------
    category: Literal["billing", "technical", "feedback", "general"]
    confidence: float          # 0.0–1.0
    detected_keywords: list[str]
    classification_reasoning: str

    # ------------------------------------------------------------------
    # Escalation — populated by the Escalation Checker
    # ------------------------------------------------------------------
    is_critical: bool
    escalation_reasons: list[str]
    contact_count_7d: int

    # ------------------------------------------------------------------
    # RAG — populated by the Retrieve Context node
    # ------------------------------------------------------------------
    retrieved_chunks: list[dict]   # Each: {content, metadata, score}
    citations: list[str]           # Source filenames

    # ------------------------------------------------------------------
    # Response — populated by the Response Drafter & Guardrail Validator
    # ------------------------------------------------------------------
    draft_response: str
    guardrail_passed: bool
    guardrail_violations: list[str]

    # ------------------------------------------------------------------
    # Routing & Audit — populated by routing nodes
    # ------------------------------------------------------------------
    final_action: Literal["send_auto_reply", "route_to_human"]
    assigned_agent_queue: str   # e.g. "billing-tier-2", "security-team", "human_review"
    audit_log: list[str]
