"""Build the search index for a docs repo.

Output layout (default ./index):
    chunks.db        SQLite: chunks table + FTS5 keyword index
    embeddings.npy   float32 [n_chunks, dim], row i == chunk id i+1
    meta.json        embedder name, dim, source dir, counts, timestamp
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import List

import numpy as np

from .chunker import Chunk, chunk_file
from .embedder import get_embedder
from .textutil import index_tokens

DOC_SUFFIXES = {".md", ".markdown", ".mdx", ".txt"}
SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", "index", ".obsidian"}


def collect_chunks(docs_dir: Path) -> List[Chunk]:
    chunks: List[Chunk] = []
    files = sorted(
        p
        for p in docs_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in DOC_SUFFIXES
        and not any(part in SKIP_DIRS or part.startswith(".") for part in p.relative_to(docs_dir).parts[:-1])
    )
    for p in files:
        chunks.extend(chunk_file(p, str(p.relative_to(docs_dir))))
    return chunks


def build_index(
    docs_dir: str | Path,
    index_dir: str | Path = "index",
    with_vectors: bool = True,
    embedder_name: str = "auto",
) -> dict:
    docs_dir = Path(docs_dir).resolve()
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    chunks = collect_chunks(docs_dir)
    if not chunks:
        raise SystemExit(f"no .md/.txt documents found under {docs_dir}")
    n_files = len({c.path for c in chunks})
    print(f"indexing {len(chunks)} chunks from {n_files} files ...", file=sys.stderr)

    db_path = index_dir / "chunks.db"
    if db_path.exists():
        db_path.unlink()
    con = sqlite3.connect(db_path)
    con.executescript(
        """
        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY,
            path TEXT NOT NULL,
            heading TEXT NOT NULL,
            start_line INTEGER NOT NULL,
            text TEXT NOT NULL
        );
        CREATE VIRTUAL TABLE fts USING fts5(tokens);
        """
    )
    for i, c in enumerate(chunks, start=1):
        con.execute(
            "INSERT INTO chunks (id, path, heading, start_line, text) VALUES (?,?,?,?,?)",
            (i, c.path, c.heading, c.start_line, c.text),
        )
        con.execute(
            "INSERT INTO fts (rowid, tokens) VALUES (?,?)",
            (i, " ".join(index_tokens(c.index_text))),
        )
    con.commit()
    con.close()

    meta = {
        "docs_dir": str(docs_dir),
        "files": n_files,
        "chunks": len(chunks),
        "embedder": None,
        "dim": None,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    emb_path = index_dir / "embeddings.npy"
    if with_vectors:
        emb = get_embedder(embedder_name)
        mat = emb.encode_docs([c.index_text for c in chunks])
        np.save(emb_path, mat.astype(np.float32))
        meta["embedder"] = emb.name
        meta["dim"] = int(mat.shape[1])
    elif emb_path.exists():
        # a stale matrix from a previous run would misalign with the new
        # chunk ids — remove it
        emb_path.unlink()

    (index_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"done in {time.time() - t0:.1f}s -> {index_dir}/", file=sys.stderr)
    return meta
