"""
Pydantic request and response schemas for the FastAPI layer.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, EmailStr, Field


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class EmailRequest(BaseModel):
    """Payload for submitting an email for processing."""

    email_id: str = Field(
        description="Unique identifier for the email. Must be unique per request."
    )
    sender_email: str = Field(
        description="Email address of the sender (customer).",
        examples=["customer@example.com"],
    )
    subject: str = Field(description="Email subject line.")
    body: str = Field(description="Full email body text.")
    received_at: Optional[str] = Field(
        default=None,
        description="ISO-8601 timestamp of when the email was received. Defaults to now.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "email_id": "em_001",
                "sender_email": "alice@example.com",
                "subject": "Refund request for my last order",
                "body": "Hi, I was charged but the service didn't work. Can I get a refund?",
                "received_at": "2026-08-05T12:00:00Z",
            }
        }
    }


class IngestRequest(BaseModel):
    """Payload for triggering a knowledge base re-ingestion."""

    docs_dir: str = Field(
        default="./knowledge_base",
        description="Path to the directory containing documents to ingest.",
    )


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class ClassificationInfo(BaseModel):
    category: Literal["billing", "technical", "feedback", "general"]
    confidence: float
    detected_keywords: list[str]
    reasoning: str


class EscalationInfo(BaseModel):
    is_critical: bool
    reasons: list[str]
    contact_count_7d: int


class GuardrailInfo(BaseModel):
    passed: bool
    violations: list[str]


class ProcessingResult(BaseModel):
    """Full result returned after an email is processed by the pipeline."""

    email_id: str
    final_action: Literal["send_auto_reply", "route_to_human"]
    assigned_agent_queue: Optional[str] = None

    # Sub-results
    classification: ClassificationInfo
    escalation: EscalationInfo
    guardrail: GuardrailInfo

    # Response
    draft_response: Optional[str] = None
    citations: list[str] = Field(default_factory=list)

    # Audit
    audit_log: list[str] = Field(default_factory=list)
    processed_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z"
    )


class IngestResult(BaseModel):
    chunks_stored: int
    message: str


class HealthResult(BaseModel):
    status: Literal["ok", "degraded"]
    redis_connected: bool
    chroma_connected: bool
