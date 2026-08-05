"""
Conversational QA Agent with Memory & Tool Use
================================================
A LangGraph-based agent that:
  1. Answers questions grounded in a sample document
  2. Maintains conversation memory across turns
  3. Selectively invokes tools (document search, calculator) when needed

Framework: LangGraph (open-source) + Google Gemini API
Usage:     python agent.py
"""

import os
import re
import math
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from langgraph.checkpoint.memory import MemorySaver

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

DOCUMENT_PATH = Path(__file__).parent / "sample_document.txt"
MODEL_NAME = "gemini-3.5-flash"

SYSTEM_PROMPT = """You are Nova, a friendly and knowledgeable assistant for Nova Dynamics Inc.

RULES:
1. When the user introduces themselves, greet them warmly by name and remember it.
2. When answering questions about the company, products, or policies, ALWAYS use the
   search_document tool first to find relevant information. Only answer based on what
   the tool returns — never invent facts.
3. When the user asks you to calculate something (totals, discounts, percentages, etc.),
   use the calculator tool. Show the calculation clearly.
4. For casual conversation (greetings, how are you, etc.), respond directly without tools.
5. If the document doesn't contain the answer, say so honestly.
6. You have full memory of this conversation — reference prior context naturally.
"""


# ---------------------------------------------------------------------------
# Tool: Document Search
# ---------------------------------------------------------------------------

def _load_document() -> str:
    """Load the sample document from disk."""
    try:
        return DOCUMENT_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return "[ERROR] sample_document.txt not found."


def _search_passages(query: str, document: str, context_lines: int = 4) -> str:
    """
    Simple keyword search: finds lines containing any query keyword,
    then returns those lines with surrounding context.
    """
    lines = document.splitlines()
    query_keywords = [w.lower() for w in query.split() if len(w) > 2]

    # Score every line by how many keywords it contains
    scored: list[tuple[int, int]] = []
    for idx, line in enumerate(lines):
        line_lower = line.lower()
        score = sum(1 for kw in query_keywords if kw in line_lower)
        if score > 0:
            scored.append((idx, score))

    if not scored:
        return "No relevant passages found in the document."

    # Take the top-scoring unique regions
    scored.sort(key=lambda x: x[1], reverse=True)
    seen_ranges: set[int] = set()
    passages: list[str] = []

    for idx, _score in scored[:5]:
        if idx in seen_ranges:
            continue
        start = max(0, idx - context_lines)
        end = min(len(lines), idx + context_lines + 1)
        seen_ranges.update(range(start, end))
        snippet = "\n".join(lines[start:end])
        passages.append(snippet)

    return "\n\n---\n\n".join(passages)


@tool
def search_document(query: str) -> str:
    """Search the Nova Dynamics company document for information relevant to the query.

    Use this tool when the user asks about company information, products,
    pricing, policies, contact details, warranties, or any factual question
    about Nova Dynamics.

    Args:
        query: The search query — keywords or a natural language question.
    """
    document = _load_document()
    if document.startswith("[ERROR]"):
        return document
    return _search_passages(query, document)


# ---------------------------------------------------------------------------
# Tool: Calculator
# ---------------------------------------------------------------------------

# Whitelist of safe names for the calculator eval
_CALC_SAFE_NAMES = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "pow": pow,
    "sqrt": math.sqrt,
    "pi": math.pi,
    "e": math.e,
}


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression and return the result.

    Use this tool when the user asks you to compute something — totals,
    discounts, percentages, unit conversions, or any arithmetic.

    Args:
        expression: A mathematical expression to evaluate, e.g. "249.99 * 3 * 0.85"
    """
    # Sanitize: only allow digits, operators, parentheses, dots, and known functions
    sanitized = expression.strip()
    if not re.match(r'^[\d\s\+\-\*/\.\(\)%,a-zA-Z_]+$', sanitized):
        return f"Error: Invalid characters in expression: {sanitized}"

    try:
        result = eval(sanitized, {"__builtins__": {}}, _CALC_SAFE_NAMES)  # noqa: S307
        # Format nicely: avoid floating point ugliness like 749.9699999999999
        if isinstance(result, float):
            result = round(result, 2)
        return f"{sanitized} = {result}"
    except Exception as exc:
        return f"Error evaluating '{sanitized}': {exc}"


# ---------------------------------------------------------------------------
# Agent Setup
# ---------------------------------------------------------------------------

def create_agent():
    """Build and return the LangGraph ReAct agent with memory."""

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY not set. Get a free key at https://aistudio.google.com\n"
            "Then create a .env file with: GOOGLE_API_KEY=your_key_here"
        )

    # Initialize the Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=api_key,
        temperature=0.3,
    )

    # Tools the agent can call
    tools = [search_document, calculator]

    # Memory: persists conversation history across turns
    memory = MemorySaver()

    # Create the ReAct agent — LangGraph handles the tool-calling loop
    agent = create_react_agent(
        model=llm,
        tools=tools,
        checkpointer=memory,
        prompt=SYSTEM_PROMPT,
    )

    return agent


# ---------------------------------------------------------------------------
# CLI Chat Loop
# ---------------------------------------------------------------------------

def main():
    """Run the interactive chat loop."""

    print("=" * 60)
    print("  Nova Dynamics AI Assistant")
    print("  Powered by LangGraph + Gemini")
    print("=" * 60)
    print("  Type your message and press Enter.")
    print("  Type 'quit' or 'exit' to end the conversation.")
    print("=" * 60)
    print()

    agent = create_agent()

    # Fixed thread config — keeps memory across all turns in this session
    config = {"configurable": {"thread_id": "user-session-1"}}

    while True:
        try:
            user_input = input("👤 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        # Invoke the agent with the user's message
        response = agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
        )

        # Extract the final assistant message and normalize content
        ai_message = response["messages"][-1]
        content = ai_message.content

        # Newer Gemini models return content as a list of blocks
        # e.g. [{'type': 'text', 'text': '...', 'extras': {...}}]
        if isinstance(content, list):
            text_parts = [
                block["text"]
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            content = "\n".join(text_parts)

        print(f"\n🤖 Nova: {content}\n")
        print("-" * 60)


if __name__ == "__main__":
    main()
