# Agentic Customer Support Email Processing System

A production-grade agentic system built with **LangGraph** and **FastAPI** that automates the classification, routing, and response drafting for customer support emails — with hard guardrails against hallucination and deterministic escalation logic.

## Features

- **Graph-based orchestration** via LangGraph StateGraph
- **Structured LLM output** for classification (Pydantic schema enforcement)
- **Deterministic escalation** rules — no LLM for safety-critical routing
- **RAG pipeline** grounded in the company knowledge base (ChromaDB)
- **Three-layer guardrail validation** before any response is sent
- **Redis-backed** contact frequency tracking (with in-memory fallback)
- **FastAPI** service layer with API key authentication

## Quick Start

```bash
# 1. Copy environment config
cp .env.example .env
# Edit .env and add your own Google Gemini API key: GOOGLE_API_KEY

# 2. Install
pip install -e ".[dev]"

# 3. Ingest knowledge base
python -m rag.ingest

# 4. Run the API server
uvicorn api.main:app --reload

# 5. Run tests
pytest tests/ -v
```

## Architecture

This system uses a **multi-agent orchestration architecture** built with LangGraph:
- **Classification Node**: Uses structured LLM parsing to categorize emails (Billing, Technical, Feedback).
- **Escalation/Routing Node**: Uses deterministic rule-based logic (e.g., keyword matching for data loss/security breach, Redis-backed frequency tracking) to route critical issues to human agents *before* any LLM drafting occurs.
- **RAG & Drafting Node**: Uses a ChromaDB-backed retrieval pipeline to ground responses in company FAQs/PDFs, featuring a three-layer guardrail validation to prevent hallucinations regarding sensitive topics like refund policies.
