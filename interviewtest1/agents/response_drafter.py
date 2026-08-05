"""
Response Drafter Agent — generates a grounded customer support reply
using only the retrieved knowledge base chunks.

Anti-hallucination constraints are baked into the system prompt.
For refund-related queries, an additional constraint forces verbatim
quoting from the refund policy document.
"""

from __future__ import annotations

import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from agents.state import EmailState
from config.settings import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

BASE_SYSTEM_PROMPT = """\
You are a professional customer support agent for a SaaS company. Your job is \
to write a helpful, polite, and concise reply to a customer support email.

STRICT RULES — you MUST follow all of them:
1. Answer ONLY using information from the provided CONTEXT below.
2. If the context does not contain enough information to answer the customer's \
question, reply with exactly: \
"I don't have enough information to answer this. Let me connect you with a \
specialist who can help you directly."
3. NEVER invent policies, prices, timelines, deadlines, contact information, \
or procedures not explicitly present in the context.
4. NEVER make up dollar amounts or dates.
5. Be warm, professional, and concise. Aim for 3–6 sentences.
6. End with: "Best regards,\\nCustomer Support Team"
"""

REFUND_ADDITIONAL_CONSTRAINT = """\

ADDITIONAL CONSTRAINT (REFUND QUERY DETECTED):
7. For any refund-related claims, you MUST quote verbatim from the provided \
refund policy context. Do not paraphrase refund timelines, eligibility \
criteria, or procedures — use the exact wording from the document.
"""

USER_TEMPLATE = """\
## Customer Email

Subject: {subject}
From: {sender_email}

{body}

---
## Retrieved Context

{context}

---
Write a reply to this customer email using ONLY the information in the context above.
"""

# ---------------------------------------------------------------------------
# Refund keyword detection (mirrors guardrails logic)
# ---------------------------------------------------------------------------

_REFUND_KEYWORDS = frozenset(
    {"refund", "money back", "return", "cancellation", "reimbursement", "reimburse"}
)


def _is_refund_email(state: EmailState) -> bool:
    text = (state.get("subject", "") + " " + state.get("body", "")).lower()
    return any(kw in text for kw in _REFUND_KEYWORDS)


def _build_context_str(chunks: list[dict]) -> str:
    if not chunks:
        return "(No relevant context was retrieved from the knowledge base.)"
    parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("metadata", {}).get("source_file", "unknown")
        parts.append(f"[{i}] (Source: {source})\n{chunk['content']}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# LLM setup
# ---------------------------------------------------------------------------

_llm = ChatGoogleGenerativeAI(
    model=settings.google_model_drafter,
    temperature=0.2,  # slight creativity for natural tone, but grounded
    google_api_key=settings.google_api_key,
)


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------

def draft_response(state: EmailState) -> EmailState:
    """
    LangGraph node: drafts a grounded customer support response.

    Reads:  email_id, sender_email, subject, body, retrieved_chunks, citations
    Writes: draft_response, audit_log
    """
    email_id = state["email_id"]
    chunks = state.get("retrieved_chunks", [])
    context_str = _build_context_str(chunks)

    # Build system prompt (add refund constraint if needed)
    system_prompt = BASE_SYSTEM_PROMPT
    if _is_refund_email(state):
        system_prompt += REFUND_ADDITIONAL_CONSTRAINT
        logger.info("[%s] Refund email detected — applying verbatim-quote constraint.", email_id)

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", USER_TEMPLATE)]
    )
    chain = prompt | _llm

    logger.info("[%s] Drafting response with %d retrieved chunks.", email_id, len(chunks))

    response = chain.invoke(
        {
            "subject": state.get("subject", ""),
            "sender_email": state.get("sender_email", ""),
            "body": state.get("body", ""),
            "context": context_str,
        }
    )

    content = response.content if hasattr(response, "content") else str(response)
    if isinstance(content, list):
        draft = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    else:
        draft = str(content)

    log_entry = f"draft_response: generated {len(draft)} chars using {len(chunks)} chunks"
    logger.info("[%s] %s", email_id, log_entry)

    return {
        **state,
        "draft_response": draft,
        "audit_log": [*state.get("audit_log", []), log_entry],
    }
