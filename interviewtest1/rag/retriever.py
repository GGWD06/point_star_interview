"""
Hybrid retrieval module.

Provides semantic search against ChromaDB with optional metadata filtering
for refund-related queries and a relevance threshold guard.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config.settings import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    chunks: list[dict] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    num_retrieved: int = 0


# ---------------------------------------------------------------------------
# Refund-query detection
# ---------------------------------------------------------------------------

REFUND_KEYWORDS = frozenset(
    {
        "refund",
        "money back",
        "return",
        "cancellation",
        "cancel",
        "reimbursement",
        "reimburse",
        "charge back",
        "chargeback",
        "get my money",
    }
)


def _is_refund_query(query: str) -> bool:
    """Heuristic: does this query concern a refund?"""
    query_lower = query.lower()
    return any(kw in query_lower for kw in REFUND_KEYWORDS)


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class KnowledgeBaseRetriever:
    """
    Wraps ChromaDB for semantic search with:
    - Metadata filtering for refund queries
    - Relevance threshold filtering
    """

    def __init__(self) -> None:
        embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.google_embedding_model,
            google_api_key=settings.google_api_key,
        )
        self._vectorstore = Chroma(
            collection_name=settings.chroma_collection_name,
            embedding_function=embeddings,
            persist_directory=settings.chroma_persist_dir,
        )

    def retrieve(self, query: str, email_body: str = "") -> RetrievalResult:
        """
        Retrieve relevant chunks for the given query.

        Args:
            query: The search query (typically email subject + first sentence of body).
            email_body: Full email body, used for refund keyword detection.

        Returns:
            RetrievalResult with filtered chunks and citations.
        """
        combined_text = f"{query} {email_body}"
        apply_refund_filter = _is_refund_query(combined_text)

        # Build search kwargs
        search_kwargs: dict = {"k": settings.rag_top_k}
        if apply_refund_filter:
            logger.info("Refund query detected — restricting to refund policy chunks.")
            search_kwargs["filter"] = {"is_refund_policy": True}

        # Similarity search with scores
        results_with_scores: list[tuple[Document, float]] = (
            self._vectorstore.similarity_search_with_relevance_scores(
                query=query,
                **search_kwargs,
            )
        )

        # Apply relevance threshold
        filtered: list[tuple[Document, float]] = [
            (doc, score)
            for doc, score in results_with_scores
            if score >= settings.rag_relevance_threshold
        ]

        if not filtered:
            logger.warning(
                "No chunks met the relevance threshold (%.2f) for query: %r",
                settings.rag_relevance_threshold,
                query,
            )

        chunks = []
        citations = []
        for doc, score in filtered:
            chunks.append(
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": round(score, 4),
                }
            )
            source = doc.metadata.get("source_file", "unknown")
            if source not in citations:
                citations.append(source)

        return RetrievalResult(
            chunks=chunks,
            citations=citations,
            num_retrieved=len(chunks),
        )


# ---------------------------------------------------------------------------
# Module-level singleton (lazy init)
# ---------------------------------------------------------------------------
_retriever: KnowledgeBaseRetriever | None = None


def get_retriever() -> KnowledgeBaseRetriever:
    global _retriever
    if _retriever is None:
        _retriever = KnowledgeBaseRetriever()
    return _retriever
