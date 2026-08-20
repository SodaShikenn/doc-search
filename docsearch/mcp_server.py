"""MCP server — makes the hybrid search a tool for Claude Code (Agentic RAG).

Claude Code is the agent/reasoning engine: it decides what to search,
reformulates queries, follows up on hits by reading the cited files, and
synthesizes the answer with citations. This server only does retrieval.

Register in the docs repo's .mcp.json:

    {
      "mcpServers": {
        "docsearch": {
          "command": "/path/to/doc-search/.venv/bin/python",
          "args": ["-m", "docsearch.mcp_server"],
          "env": { "DOCSEARCH_INDEX": "/path/to/doc-search/index" }
        }
      }
    }
"""

from __future__ import annotations

import os
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from .envutil import load_dotenv
from .search import SearchIndex

load_dotenv()

mcp = MCPServer(
    "docsearch",
    instructions=(
        "Hybrid search over the team's documentation repo (glossary, review "
        "perspectives, architecture docs; Japanese + English). Agentic-RAG "
        "loop: (1) search_docs with mode='hybrid'; (2) if hits are weak, "
        "reformulate — exact terms via mode='keyword', paraphrases or "
        "natural-language questions via mode='vector'; (3) Read the cited "
        "file for full context; (4) answer citing path:line."
    ),
)

_index: SearchIndex | None = None


def _get_index() -> SearchIndex:
    global _index
    if _index is None:
        _index = SearchIndex(os.environ.get("DOCSEARCH_INDEX", "index"))
    return _index


def _format(out: dict) -> str:
    if not out["results"]:
        return (
            "No hits. Try: (1) mode='vector' with a paraphrase or a full "
            "question, (2) fewer/other keywords, (3) the English or Japanese "
            "synonym of the term."
        )
    lines = []
    for i, r in enumerate(out["results"], start=1):
        via = []
        if r.get("kw_rank"):
            via.append(f"kw#{r['kw_rank']}")
        if r.get("vec_rank"):
            via.append(f"vec#{r['vec_rank']}:{r['vec_score']:.3f}")
        head = f" | {r['heading']}" if r["heading"] else ""
        lines.append(
            f"[{i}] {r['path']}:{r['line']} ({','.join(via)}){head}\n    {r['snippet']}"
        )
    lines.append(
        "\nTip: results are chunks. For full context, read the file at the "
        "cited path (relative to the indexed docs repo) before answering, and "
        "cite path:line in your answer."
    )
    return "\n".join(lines)


@mcp.tool()
def search_docs(query: str, mode: str = "hybrid", k: int = 8) -> str:
    """Search the project's documentation repo (glossary, review perspectives,
    architecture docs). Supports Japanese and English.

    mode:
      - "hybrid"  (default) keyword BM25 + semantic vectors, fused by RRF
      - "keyword" exact-term match (best for identifiers / exact terms)
      - "vector"  semantic match (best for paraphrases or natural-language
        questions, e.g. 「解約率」 finds チャーンレート)

    Returns ranked chunks as `path:line | heading` plus a snippet. Follow up
    by reading the cited file for full context; cite path:line in answers.
    """
    if mode not in ("hybrid", "keyword", "vector"):
        return "mode must be one of: hybrid, keyword, vector"
    out = _get_index().search(query, mode=mode, k=max(1, min(k, 20)))
    return _format(out)


@mcp.tool()
def docs_repo_info() -> str:
    """Show what documentation repo is indexed (path, file/chunk counts,
    embedder). Use the docs_dir as the base for reading cited files."""
    idx = _get_index()
    m = idx.meta
    return (
        f"docs_dir: {m['docs_dir']}\nfiles: {m['files']}  chunks: {m['chunks']}\n"
        f"embedder: {m.get('embedder') or 'none (keyword only)'}\n"
        f"built_at: {m['built_at']}\n"
        f"index_dir: {Path(idx.index_dir).resolve()}"
    )


def main() -> None:
    mcp.run()  # stdio transport


if __name__ == "__main__":
    main()
