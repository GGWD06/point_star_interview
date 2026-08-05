"""
pytest configuration and shared fixtures.

Sets OPENAI_API_KEY to a dummy value so that Settings() doesn't fail
during test collection (no real API calls are made — everything is mocked).
"""

import os
import pytest

# Prevent pydantic-settings from complaining about missing env vars during tests
os.environ.setdefault("GOOGLE_API_KEY", "test-dummy-key-for-testing-only")
os.environ.setdefault("REDIS_URL", "")
os.environ.setdefault("CHROMA_PERSIST_DIR", "./test_chroma_db")
