"""Query the index: keyword (FTS5/BM25), vector (cosine), hybrid (RRF).

Reciprocal Rank Fusion: score(d) = Σ_lists 1 / (K + rank_d), K = 60.
Rank-based fusion avoids having to calibrate BM25 scores against cosine
similarities, which live on incomparable scales.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .embedder import get_embedder
from .gitlink import doc_url
from .textutil import fts_query, normalize, query_terms

RRF_K = 60
CANDIDATES = 50  # depth of each list fed into the fusion


class SearchIndex:
    def __init__(self, index_dir: str | Path = "index"):
        self.index_dir = Path(index_dir)
        meta_path = self.index_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"no index at {self.index_dir}/ — run: python -m docsearch index <docs_dir>"
            )
        self.meta = json.loads(meta_path.read_text())
        self.con = sqlite3.connect(self.index_dir / "chunks.db", check_same_thread=False)
        self._db_lock = threading.Lock()
        emb_path = self.index_dir / "embeddings.npy"
        self.embeddings: Optional[np.ndarray] = None
        if emb_path.exists() and self.meta.get("embedder"):
            mat = np.load(emb_path)
            # a matrix misaligned with chunks.db would silently return wrong
            # documents — refuse it
            if len(mat) == self.meta.get("chunks"):
                self.embeddings = mat
        self._embedder = None
        self._emb_lock = threading.Lock()

    # -- backends -----------------------------------------------------------

    def ensure_embedder(self):
        with self._emb_lock:
            if self._embedder is None:
                self._embedder = get_embedder(self.meta.get("embedder") or "auto")
            return self._embedder

    def keyword(self, query: str, k: int = CANDIDATES) -> List[Dict]:
        expr = fts_query(query)
        if not expr:
            return []
        with self._db_lock:
            rows = self.con.execute(
                "SELECT rowid, bm25(fts) FROM fts WHERE fts MATCH ? "
                "ORDER BY bm25(fts) LIMIT ?",
                (expr, k),
            ).fetchall()
        # sqlite bm25(): smaller is better; negate so bigger is better
        return [
            {"id": r[0], "rank": i + 1, "score": -float(r[1])}
            for i, r in enumerate(rows)
        ]

    def vector(self, query: str, k: int = CANDIDATES) -> List[Dict]:
        if self.embeddings is None:
            return []
        q = self.ensure_embedder().encode_query(query)
        sims = self.embeddings @ q
        k = min(k, len(sims))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [
            {"id": int(i) + 1, "rank": r + 1, "score": float(sims[i])}
            for r, i in enumerate(top)
        ]

    # -- fusion + presentation ----------------------------------------------

    def search(self, query: str, mode: str = "hybrid", k: int = 10, include_text: bool = False) -> Dict:
        query = query.strip()
        if not query:
            return {"query": query, "mode": mode, "results": []}

        kw = self.keyword(query) if mode in ("hybrid", "keyword") else []
        vec = self.vector(query) if mode in ("hybrid", "vector") else []

        fused: Dict[int, Dict] = {}
        for hit in kw:
            fused.setdefault(hit["id"], {"id": hit["id"], "rrf": 0.0})
            fused[hit["id"]]["kw_rank"] = hit["rank"]
            fused[hit["id"]]["kw_score"] = hit["score"]
            fused[hit["id"]]["rrf"] += 1.0 / (RRF_K + hit["rank"])
        for hit in vec:
            fused.setdefault(hit["id"], {"id": hit["id"], "rrf": 0.0})
            fused[hit["id"]]["vec_rank"] = hit["rank"]
            fused[hit["id"]]["vec_score"] = hit["score"]
            fused[hit["id"]]["rrf"] += 1.0 / (RRF_K + hit["rank"])

        ranked = sorted(fused.values(), key=lambda h: -h["rrf"])[:k]
        terms = query_terms(query)
        results = []
        for h in ranked:
            with self._db_lock:
                row = self.con.execute(
                    "SELECT path, heading, start_line, text FROM chunks WHERE id = ?",
                    (h["id"],),
                ).fetchone()
            if row is None:
                continue
            path, heading, line, text = row
            item = {
                **h,
                "path": path,
                "heading": heading,
                "line": line,
                "url": doc_url(self.meta.get("github_base"), path, line),
                "snippet": _snippet(text, terms),
            }
            if include_text:
                item["text"] = text
            results.append(item)
        return {
            "query": query,
            "mode": mode,
            "terms": terms,
            "results": results,
            "keyword_hits": len(kw),
            "vector_hits": len(vec),
        }


def _earliest(haystack: str, terms: List[str]) -> int:
    pos = -1
    for t in terms:
        p = haystack.find(t)
        if p != -1 and (pos == -1 or p < pos):
            pos = p
    return pos


def _norm_to_orig(text: str, pos_n: int) -> int:
    """Map an offset in normalize(text) back to an offset in text.

    NFKC can change lengths (half-width katakana, full-width latin), so we
    scan for the first prefix whose normalized length exceeds pos_n.
    len(normalize(prefix)) is non-decreasing as the prefix grows.
    """
    for i in range(len(text)):
        if len(normalize(text[: i + 1])) > pos_n:
            return i
    return 0


def _snippet(text: str, terms: List[str], width: int = 220) -> str:
    """Window around the first query-term occurrence (or the chunk head)."""
    # fast path: exact offsets in the original (guard: lower() rarely changes
    # length, e.g. U+0130 — fall through to the mapped path if it does)
    hay = text.lower()
    pos = _earliest(hay, terms) if len(hay) == len(text) else -1
    if pos == -1:
        # match exists only after NFKC (e.g. half-width katakana) — find in
        # normalized space, then map the offset back
        pos_n = _earliest(normalize(text), terms)
        pos = _norm_to_orig(text, pos_n) if pos_n > 0 else 0
    start = max(0, pos - width // 3)
    end = min(len(text), start + width)
    snip = text[start:end].replace("\n", " ")
    prefix = "… " if start > 0 else ""
    suffix = " …" if end < len(text) else ""
    return f"{prefix}{snip}{suffix}"
