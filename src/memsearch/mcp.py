"""
MemSearch MCP Server — lightweight, auto-initializing memory for AI agents.

Usage (Agent Studio):
  {
    "mcpServers": {
      "memsearch": {
        "command": "uvx",
        "args": ["--from", "memsearch[mcp]", "memsearch-mcp"],
        "env": {
          "MEMSEARCH_MEMORY_PATH": "/home/cdsw/.memsearch/memories",
          "MEMSEARCH_COLLECTION": "my_workflow",
          "MEMSEARCH_EMBEDDING_PROVIDER": "openai",
          "MEMSEARCH_EMBEDDING_API_KEY": "${OPENAI_API_KEY}"
        }
      }
    }
  }

Usage (Claude Code):
  claude mcp add memsearch -- memsearch-mcp

Environment Variables
---------------------
MEMSEARCH_MEMORY_PATH         Directory for dated markdown memory files (auto-created).
                              Default: ~/.memsearch/memories
MEMSEARCH_MILVUS_URI          Milvus database path/URL. Default: {MEMORY_PATH}/milvus.db
MEMSEARCH_COLLECTION          Collection name (unique per workflow). Default: memsearch_memory
MEMSEARCH_EMBEDDING_PROVIDER  openai | ollama | local | onnx | google | voyage
MEMSEARCH_EMBEDDING_MODEL     Override default model for the provider.
MEMSEARCH_EMBEDDING_API_KEY   API key (required for openai/google/voyage).
MEMSEARCH_EMBEDDING_BASE_URL  OpenAI-compatible endpoint base URL.
MEMSEARCH_MAX_CHUNK_SIZE      Max chars per indexed chunk. Default: 1500
MEMSEARCH_OVERLAP_LINES       Overlap lines between chunks. Default: 2

No initialization required — the first write_memory call creates everything.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from .core import MemSearch

# ---------------------------------------------------------------------------
# Configuration (all from environment variables)
# ---------------------------------------------------------------------------

_MEMORY_PATH = Path(os.environ.get("MEMSEARCH_MEMORY_PATH", str(Path.home() / ".memsearch" / "memories")))

_MILVUS_URI = os.environ.get("MEMSEARCH_MILVUS_URI", str(_MEMORY_PATH / "milvus.db"))

_COLLECTION = os.environ.get("MEMSEARCH_COLLECTION", "memsearch_memory")

_EMBEDDING_PROVIDER = os.environ.get("MEMSEARCH_EMBEDDING_PROVIDER", "openai")
_EMBEDDING_MODEL = os.environ.get("MEMSEARCH_EMBEDDING_MODEL") or None
_EMBEDDING_API_KEY = os.environ.get("MEMSEARCH_EMBEDDING_API_KEY") or os.environ.get("OPENAI_API_KEY") or None
_EMBEDDING_BASE_URL = os.environ.get("MEMSEARCH_EMBEDDING_BASE_URL") or None
_MAX_CHUNK_SIZE = int(os.environ.get("MEMSEARCH_MAX_CHUNK_SIZE", "1500"))
_OVERLAP_LINES = int(os.environ.get("MEMSEARCH_OVERLAP_LINES", "2"))

# ---------------------------------------------------------------------------
# Singleton MemSearch instance (lazy-initialized)
# ---------------------------------------------------------------------------

_mem_instance: MemSearch | None = None


def _get_mem() -> MemSearch:
    global _mem_instance
    if _mem_instance is None:
        _MEMORY_PATH.mkdir(parents=True, exist_ok=True)

        _mem_instance = MemSearch(
            paths=[str(_MEMORY_PATH)],
            embedding_provider=_EMBEDDING_PROVIDER,
            embedding_model=_EMBEDDING_MODEL,
            embedding_api_key=_EMBEDDING_API_KEY,
            embedding_base_url=_EMBEDDING_BASE_URL,
            milvus_uri=_MILVUS_URI,
            collection=_COLLECTION,
            max_chunk_size=_MAX_CHUNK_SIZE,
            overlap_lines=_OVERLAP_LINES,
        )
    return _mem_instance


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "memsearch-memory",
    instructions=(
        "Memory tools backed by dated markdown files and hybrid vector+BM25 search. "
        "Use write_memory to store structured notes and search_memory to recall them "
        "across sessions. Prefer compact, structured content over verbatim conversation text."
    ),
)


# ---------------------------------------------------------------------------
# Tool 1: get_current_date
# ---------------------------------------------------------------------------


@mcp.tool()
def get_current_date() -> str:
    """
    Return the current date and time in ISO 8601 format (YYYY-MM-DDTHH:MM:SS).

    Call this before write_memory when you need a precise timestamp for the heading.
    """
    return datetime.now().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Tool 2: write_memory
# ---------------------------------------------------------------------------


@mcp.tool()
async def write_memory(content: str, heading: str | None = None) -> dict[str, Any]:
    """
    Write a memory entry to today's dated markdown file and index it immediately.

    The entry is appended to {MEMORY_PATH}/YYYY-MM-DD.md under an H2 heading.
    The vector index is updated in-place — no separate indexing step needed.

    Best practices:
    - Store a STRUCTURED NOTE, not raw conversation text.
    - Include key entities: customer IDs, account numbers, transaction IDs.
    - One write_memory call per significant interaction turn, not per message.
    - Use a descriptive heading (e.g. "ACCOUNT_STATUS - Maria Garcia").

    Args:
        content: The memory content. Should be a compact structured note.
        heading: Optional H2 heading. Defaults to current HH:MM:SS timestamp.

    Returns:
        dict with file path, heading used, and chunks indexed count.
    """
    mem = _get_mem()
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    heading_str = heading or now.strftime("%H:%M:%S")

    file_path = _MEMORY_PATH / f"{date_str}.md"
    entry = f"\n## {heading_str}\n\n{content}\n"

    with open(file_path, "a", encoding="utf-8") as f:
        f.write(entry)

    chunks_indexed = await mem.index_file(str(file_path))

    return {
        "file": str(file_path),
        "heading": heading_str,
        "chunks_indexed": chunks_indexed,
    }


# ---------------------------------------------------------------------------
# Tool 3: search_memory
# ---------------------------------------------------------------------------


@mcp.tool()
async def search_memory(
    query: str,
    top_k: int = 5,
    source_date_prefix: str | None = None,
) -> list[dict[str, Any]]:
    """
    Search memory using hybrid dense-vector + BM25 retrieval with RRF reranking.

    Returns the most relevant memory chunks matching the query.

    Args:
        query: Natural language search query.
        top_k: Maximum number of results to return (default 5).
        source_date_prefix: Optional date filter (e.g. "2026-03-24") to restrict
                           results to memories from a specific day.

    Returns:
        List of dicts with: content, source, heading, score.
    """
    mem = _get_mem()

    source_prefix = None
    if source_date_prefix:
        source_prefix = str(_MEMORY_PATH / source_date_prefix)

    results = await mem.search(query, top_k=top_k, source_prefix=source_prefix)

    return [
        {
            "content": r.get("content", ""),
            "source": r.get("source", ""),
            "heading": r.get("heading", ""),
            "score": round(r.get("score", 0.0), 4),
        }
        for r in results
    ]


# ---------------------------------------------------------------------------
# Tool 4: list_memory_files
# ---------------------------------------------------------------------------


@mcp.tool()
def list_memory_files() -> list[dict[str, str]]:
    """
    List all dated markdown memory files in the memory directory.

    Returns a list of files sorted newest-first with name, path, and size.
    Useful for understanding what date ranges have stored memories.
    """
    if not _MEMORY_PATH.exists():
        return []

    files = sorted(_MEMORY_PATH.glob("*.md"), reverse=True)
    output = []
    for f in files:
        stat = f.stat()
        output.append({
            "name": f.name,
            "path": str(f),
            "size_bytes": str(stat.st_size),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        })
    return output


# ---------------------------------------------------------------------------
# Tool 5: reindex_memory
# ---------------------------------------------------------------------------


@mcp.tool()
async def reindex_memory() -> dict[str, int]:
    """
    Rebuild the vector index from all markdown files in the memory directory.

    Call this if you've manually edited or deleted memory files outside the MCP,
    or if search results seem stale. Safe to call at any time — idempotent.

    Returns:
        dict with total files processed and total chunks indexed.
    """
    mem = _get_mem()
    total_chunks = await mem.index(force=True)

    return {
        "files_processed": len(list(_MEMORY_PATH.glob("*.md"))),
        "total_chunks_indexed": total_chunks,
    }


# ---------------------------------------------------------------------------
# Tool 6: show_config
# ---------------------------------------------------------------------------


@mcp.tool()
def show_config() -> dict[str, str]:
    """
    Show the resolved configuration for this MemSearch instance.

    Useful for debugging connectivity or verifying which embedding
    provider is active. API keys are masked.
    """
    masked_key = (
        f"{_EMBEDDING_API_KEY[:6]}...{_EMBEDDING_API_KEY[-4:]}"
        if _EMBEDDING_API_KEY and len(_EMBEDDING_API_KEY) > 10
        else ("(not set)" if not _EMBEDDING_API_KEY else "(set)")
    )
    return {
        "memory_path": str(_MEMORY_PATH),
        "milvus_uri": _MILVUS_URI,
        "collection": _COLLECTION,
        "embedding_provider": _EMBEDDING_PROVIDER,
        "embedding_model": _EMBEDDING_MODEL or "(provider default)",
        "embedding_api_key": masked_key,
        "embedding_base_url": _EMBEDDING_BASE_URL or "(not set)",
        "max_chunk_size": str(_MAX_CHUNK_SIZE),
        "overlap_lines": str(_OVERLAP_LINES),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
