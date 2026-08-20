"""Pluggable embedding backends.

The heavy ML stack (torch + sentence-transformers) is OPTIONAL — the core
install has no ML dependencies. Pick the backend per index with
`docsearch index ... --embedder NAME`; the name is stored in meta.json and
reused automatically at query time.

| name       | where  | weight            | notes                                   |
|------------|--------|-------------------|-----------------------------------------|
| voyage     | cloud  | zero local deps   | voyage-3.5 (Anthropic's recommended     |
|            |        |                   | embeddings partner); needs VOYAGE_API_KEY|
| e5         | local  | ~470MB + torch    | multilingual-e5-small, JA+EN            |
| e5-large   | local  | ~2.2GB + torch    | multilingual-e5-large, higher quality   |
| bge-m3     | local  | ~2.3GB + torch    | BAAI/bge-m3, strongest local multilingual|
| hash       | local  | zero deps         | lexical n-gram hashing (degraded)       |

Cloud backends send document/query text to the provider — only use them for
docs your team allows leaving the machine.
"""

from __future__ import annotations

import hashlib
import os
import sys
from typing import List

import numpy as np

from .textutil import index_tokens

VOYAGE_MODEL = "voyage-3.5"
VOYAGE_URL = "https://api.voyageai.com/v1/embeddings"


class STEmbedder:
    """sentence-transformers backend (torch). Optional dependency."""

    def __init__(self, name: str, model_id: str, query_prefix: str = "", doc_prefix: str = ""):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError(
                f"embedder '{name}' needs the optional local ML stack: "
                "pip install -r requirements-local.txt"
            ) from e
        self.name = name
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix
        print(f"loading embedding model {model_id} ...", file=sys.stderr)
        self.model = SentenceTransformer(model_id)

    def encode_docs(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(
            [self.doc_prefix + t for t in texts],
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 32,
        )
        return np.asarray(vecs, dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        vec = self.model.encode([self.query_prefix + text], normalize_embeddings=True)
        return np.asarray(vec, dtype=np.float32)[0]


class VoyageEmbedder:
    """Voyage AI embeddings over plain HTTPS (httpx ships with the anthropic SDK)."""

    name = "voyage"

    def __init__(self) -> None:
        self.api_key = os.environ.get("VOYAGE_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "VOYAGE_API_KEY is not set — add it to doc-search/.env "
                "or choose a local embedder (--embedder e5)"
            )

    def _embed(self, texts: List[str], input_type: str) -> np.ndarray:
        import httpx

        out: List[List[float]] = []
        for i in range(0, len(texts), 96):
            batch = texts[i : i + 96]
            resp = httpx.post(
                VOYAGE_URL,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": VOYAGE_MODEL, "input": batch, "input_type": input_type},
                timeout=60.0,
            )
            resp.raise_for_status()
            data = sorted(resp.json()["data"], key=lambda d: d["index"])
            out.extend(d["embedding"] for d in data)
        mat = np.asarray(out, dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms

    def encode_docs(self, texts: List[str]) -> np.ndarray:
        return self._embed(texts, "document")

    def encode_query(self, text: str) -> np.ndarray:
        return self._embed([text], "query")[0]


class HashEmbedder:
    """Hashing-trick bag of tokens + char trigrams. Deterministic, no deps."""

    name = "hash"
    dim = 384

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        feats = list(index_tokens(text))
        joined = "".join(feats)
        feats += [joined[i : i + 3] for i in range(len(joined) - 2)]
        for f in feats:
            h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "big")
            v[h % self.dim] += 1.0 if (h >> 32) & 1 else -1.0
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def encode_docs(self, texts: List[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts])

    def encode_query(self, text: str) -> np.ndarray:
        return self._vec(text)


_E5_PREFIXES = {"query_prefix": "query: ", "doc_prefix": "passage: "}

_REGISTRY = {
    "voyage": lambda: VoyageEmbedder(),
    "e5": lambda: STEmbedder("e5", "intfloat/multilingual-e5-small", **_E5_PREFIXES),
    "e5-large": lambda: STEmbedder("e5-large", "intfloat/multilingual-e5-large", **_E5_PREFIXES),
    "bge-m3": lambda: STEmbedder("bge-m3", "BAAI/bge-m3"),
    "hash": lambda: HashEmbedder(),
}

EMBEDDER_NAMES = ["auto", *_REGISTRY.keys()]


def get_embedder(name: str = "auto"):
    """auto: voyage if a key is set, else local e5, else hash (with warning)."""
    if name != "auto":
        return _REGISTRY[name]()
    if os.environ.get("VOYAGE_API_KEY"):
        return VoyageEmbedder()
    try:
        return _REGISTRY["e5"]()
    except RuntimeError:
        print(
            "no VOYAGE_API_KEY and no local ML stack — falling back to hash "
            "embedder (no semantic search). Set a key or pip install -r "
            "requirements-local.txt",
            file=sys.stderr,
        )
        return HashEmbedder()
