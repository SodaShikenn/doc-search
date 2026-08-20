"""The single replaceable seam for WHERE the documents live.

This machine ships only a placeholder (sample_docs). The desktop that has
access to the real docs repo plugs it in WITHOUT code changes — resolution
order, highest priority first:

1. Environment variables (what Docker uses):
     DOCSEARCH_DOCS         path to the docs repo/folder
     DOCSEARCH_GITHUB_BASE  blob-URL base for citation links
     DOCSEARCH_EMBEDDER     embedder name (auto/voyage/e5/...)
2. datasource.json at the repo root — copy datasource.example.json and edit.
   The file is gitignored on purpose: pointers to internal company repos
   never end up on GitHub.
3. Built-in placeholder: sample_docs/ with git auto-detected links.

Everything else (indexer, CLI, server) asks get_datasource() and never
hardcodes a location.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "datasource.json"


@dataclass
class DataSource:
    name: str
    docs_dir: str
    github_base: Optional[str] = None  # None -> auto-detect from git remote
    embedder: str = "auto"


PLACEHOLDER = DataSource(
    name="sample_docs (placeholder — replace on the machine with data access)",
    docs_dir=str(PROJECT_ROOT / "sample_docs"),
)


def get_datasource(config_path: Optional[Path] = None) -> DataSource:
    ds = PLACEHOLDER

    path = config_path or CONFIG_PATH
    if path.is_file():
        cfg = json.loads(path.read_text(encoding="utf-8"))
        ds = DataSource(
            name=cfg.get("name", path.name),
            docs_dir=str(cfg["docs_dir"]),
            github_base=cfg.get("github_base") or None,
            embedder=cfg.get("embedder", "auto"),
        )

    # env vars override individual fields (Docker path)
    env_docs = os.environ.get("DOCSEARCH_DOCS")
    env_base = os.environ.get("DOCSEARCH_GITHUB_BASE")
    env_emb = os.environ.get("DOCSEARCH_EMBEDDER")
    if env_docs or env_base or env_emb:
        ds = DataSource(
            name=ds.name if not env_docs else f"env:{env_docs}",
            docs_dir=env_docs or ds.docs_dir,
            github_base=env_base or ds.github_base,
            embedder=env_emb or ds.embedder,
        )
    return ds
