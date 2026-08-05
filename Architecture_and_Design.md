# Agentic Architect Challenge - Architecture & Design Document

## 1. System Architecture
The solution is divided into three modular components addressing different agentic capabilities:
*   **Part 1: Customer Support Automation (LangGraph + FastAPI)**
    A robust state-machine architecture orchestrating email processing. It utilizes structured LLM parsing for classification (Billing, Tech, Feedback) and a ChromaDB-backed RAG pipeline for knowledge retrieval. Crucially, the routing node precedes the drafting node, ensuring safety-critical routing bypasses LLM inference.
*   **Part 2: Web Scraping Engine (Playwright + Pydantic Guardrails)**
    A hybrid extraction pipeline. It combines Playwright's headless browser capabilities to execute JavaScript and resolve single-page applications, with BeautifulSoup for HTML noise reduction. It processes the text through a Gemini-1.5-flash model bound to a Pydantic schema (`ConciseSummary`).
*   **Part 3: Conversational QA Agent (ReAct Pattern + Gemini)**
    A LangGraph-based interactive agent ("Nova") equipped with conversation memory. It dynamically leverages function calling (Tools: Document Search, Calculator) only when necessitated by the user's prompt, grounded by a single company knowledge document.

## 2. Trade-offs Made
*   **Deterministic Escalation vs. LLM Routing (Part 1)**: For routing critical issues (data loss, outages, >3 contacts), deterministic rule-based Python logic (e.g., keyword matching, Redis counters) was chosen over LLM decision-making. *Trade-off*: Sacrifices "smart" semantic routing for 100% reliability, zero-latency execution, and zero risk of hallucination on safety-critical triggers.
*   **Full Context vs. Chunking (Part 2)**: Instead of complex semantic chunking and map-reduce summarization, the system leverages Gemini 1.5's massive context window (clipping safely at 100k chars). *Trade-off*: Increases token consumption slightly but significantly reduces architectural complexity and preserves the global context of the article.
*   **Playwright vs. HTTP Requests (Part 2)**: *Trade-off*: Introduces higher latency and memory overhead compared to `requests`/`aiohttp`, but it was a necessary compromise to eliminate the "blank page" bottleneck caused by client-side rendered websites.

## 3. Addressing Potential Failure Points
*   **Mitigating Hallucinations**: In Part 1, hallucination around refund policies is prevented via a RAG pipeline with a strict system prompt ("Do not answer outside the provided context"). Additionally, a post-generation guardrail validation checks the drafted response against the source document.
*   **Formatting and Constraint Violations**: In Part 2, the LLM failing to adhere to the "<50 words" constraint is handled natively via Pydantic `@field_validator`. If the model outputs a lengthy summary, the code raises a `ValueError`, which allows the framework to automatically retry with the error context or fail gracefully without breaking the downstream pipeline.
*   **Memory Leaks in Conversational Agents**: In Part 3, the ReAct agent manages state size by maintaining a sliding window of conversation history, preventing the context payload from exceeding API limits during long sessions.

## 4. Local Environment Setup & Instructions

### Prerequisites
*   Python 3.10+
*   Google Gemini API Key (Get it free from [Google AI Studio](https://aistudio.google.com/))

### Installation & Execution
**For Part 1 (Support Email API):**
1. Navigate to `interviewtest1/`
2. Create environment file: `cp .env.example .env` and insert your `GOOGLE_API_KEY`.
3. Install dependencies: `pip install -e ".[dev]"`
4. Ingest knowledge base: `python -m rag.ingest`
5. Run server: `uvicorn api.main:app --reload`

**For Part 2 (Scraping Script):**
1. Navigate to `interviewtest2/`
2. Install dependencies: `pip install playwright bs4 langchain-google-genai pydantic`
3. Install browser binaries: `playwright install chromium`
4. Run script: `python test.py`

**For Part 3 (QA Agent):**
1. Navigate to `test/`
2. Configure `.env`: `cp .env.example .env` and insert your `GOOGLE_API_KEY`.
3. Install dependencies: `pip install -r requirements.txt`
4. Run agent CLI: `python agent.py`
