"""
Document ingestion pipeline.

Loads documents from the knowledge_base/ directory, chunks them,
embeds them with OpenAI, and persists to ChromaDB.

Usage:
    python -m rag.ingest
    python -m rag.ingest --docs-dir ./knowledge_base
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REFUND_POLICY_FILENAME = "refund_policy.md"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_pdf(path: Path) -> list[Document]:
    loader = PyPDFLoader(str(path))
    return loader.load()


def _load_markdown(path: Path) -> list[Document]:
    loader = UnstructuredMarkdownLoader(str(path), mode="elements")
    return loader.load()


def _load_document(path: Path) -> list[Document]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path)
    elif suffix in {".md", ".markdown"}:
        return _load_markdown(path)
    else:
        logger.warning("Unsupported file type: %s — skipping.", path)
        return []


# ---------------------------------------------------------------------------
# Metadata tagging
# ---------------------------------------------------------------------------

def _tag_documents(docs: list[Document], source_path: Path) -> list[Document]:
    """Attach source metadata to every chunk/document."""
    is_refund_policy = source_path.name == REFUND_POLICY_FILENAME
    document_type = "pdf" if source_path.suffix.lower() == ".pdf" else "markdown"

    for doc in docs:
        doc.metadata.update(
            {
                "source_file": source_path.name,
                "source_path": str(source_path),
                "document_type": document_type,
                "is_refund_policy": is_refund_policy,
            }
        )
    return docs


# ---------------------------------------------------------------------------
# Ingestion pipeline
# ---------------------------------------------------------------------------

def ingest(docs_dir: str | Path = "./knowledge_base") -> int:
    """
    Ingest all supported documents under *docs_dir* into ChromaDB.

    Returns the total number of chunks stored.
    """
    docs_dir = Path(docs_dir)
    if not docs_dir.exists():
        raise FileNotFoundError(f"Documents directory not found: {docs_dir}")

    # --- Collect all supported files ---
    supported_patterns = ["**/*.pdf", "**/*.md", "**/*.markdown"]
    all_paths: list[Path] = []
    for pattern in supported_patterns:
        all_paths.extend(docs_dir.glob(pattern))

    if not all_paths:
        logger.warning("No supported documents found in %s", docs_dir)
        return 0

    logger.info("Found %d document(s) to ingest.", len(all_paths))

    # --- Load & tag documents ---
    raw_docs: list[Document] = []
    for path in all_paths:
        logger.info("Loading: %s", path)
        loaded = _load_document(path)
        tagged = _tag_documents(loaded, path)
        raw_docs.extend(tagged)

    # --- Chunk ---
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)
    logger.info("Split into %d chunks.", len(chunks))

    # --- Embed & persist ---
    embeddings = GoogleGenerativeAIEmbeddings(
        model=settings.google_embedding_model,
        google_api_key=settings.google_api_key,
    )

    vectorstore = Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=embeddings,
        persist_directory=settings.chroma_persist_dir,
    )
    vectorstore.add_documents(chunks)

    logger.info(
        "Ingested %d chunks into collection '%s' at '%s'.",
        len(chunks),
        settings.chroma_collection_name,
        settings.chroma_persist_dir,
    )
    return len(chunks)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")

    parser = argparse.ArgumentParser(description="Ingest documents into ChromaDB.")
    parser.add_argument(
        "--docs-dir",
        default="./knowledge_base",
        help="Root directory containing documents to ingest (default: ./knowledge_base)",
    )
    args = parser.parse_args()

    total = ingest(args.docs_dir)
    print(f"\n✅ Ingestion complete. {total} chunks stored.")
