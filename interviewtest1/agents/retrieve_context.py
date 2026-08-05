"""
Retrieve Context node — fetches relevant chunks from the knowledge base.

This is a thin wrapper around rag.retriever that fits the LangGraph node interface.
"""

from __future__ import annotations

import logging

from agents.state import EmailState
from rag.retriever import get_retriever

logger = logging.getLogger(__name__)


def retrieve_context(state: EmailState) -> EmailState:
    """
    LangGraph node: retrieves relevant knowledge base chunks.

    Reads:  email_id, subject, body
    Writes: retrieved_chunks, citations, audit_log
    """
    email_id = state["email_id"]
    query = state.get("subject", "") + " " + state.get("body", "")[:300]

    logger.info("[%s] Retrieving context for query (first 100 chars): %r", email_id, query[:100])

    retriever = get_retriever()
    result = retriever.retrieve(query=state.get("subject", ""), email_body=state.get("body", ""))

    log_entry = (
        f"retrieve_context: retrieved {result.num_retrieved} chunks "
        f"from: {result.citations}"
    )
    logger.info("[%s] %s", email_id, log_entry)

    return {
        **state,
        "retrieved_chunks": result.chunks,
        "citations": result.citations,
        "audit_log": [*state.get("audit_log", []), log_entry],
    }
