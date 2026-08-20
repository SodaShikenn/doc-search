"""Effort-based retrieval — hides IR jargon behind an easy/medium/hard/auto dial.

Users shouldn't need to know what "hybrid vs BM25 vs vector" means. They pick
how hard the system should look (or let `auto` decide), and the retriever
reports back, in plain language, what it did and why — that explanation is
shown in the chat UI next to the sources.

  easy    one hybrid search, small context. Fast; for direct term lookups.
  medium  the standard hybrid search (previous default behavior).
  hard    an LLM generates query reformulations (JA/EN synonyms, glossary
          wording); every variant is searched and the rank lists are fused
          with RRF. For questions whose wording won't match the docs.
  auto    probe once, then escalate/de-escalate from retrieval-confidence
          signals (keyword hit strength, vector cosine, rank agreement).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from .textutil import normalize

EFFORTS = ("auto", "easy", "medium", "hard")

_K = {"easy": 4, "medium": 6, "hard": 10}
_RRF_K = 60

# auto-escalation thresholds (cosine scores from normalized embeddings)
_STRONG_COS = 0.86
_WEAK_COS = 0.80

LABELS_JA = {"easy": "かんたん", "medium": "ふつう", "hard": "しっかり"}


def _fuse(rank_lists: List[List[Dict]], k: int) -> List[Dict]:
    """Cross-list RRF over result lists (each already ranked best-first)."""
    scored: Dict[int, Dict] = {}
    for results in rank_lists:
        for rank, r in enumerate(results, start=1):
            entry = scored.setdefault(r["id"], {"item": r, "score": 0.0})
            entry["score"] += 1.0 / (_RRF_K + rank)
    ranked = sorted(scored.values(), key=lambda e: -e["score"])
    return [e["item"] for e in ranked[:k]]


def retrieve(
    index,
    query: str,
    effort: str = "auto",
    expander: Optional[Callable[[str], List[str]]] = None,
) -> Dict:
    """Returns {sources, effort_requested, effort_used, reason, queries}."""
    requested = effort if effort in EFFORTS else "auto"
    used = requested
    reason = ""
    queries = [query]

    probe = index.search(query, mode="hybrid", k=_K["medium"], include_text=True)
    results = probe["results"]

    if requested == "auto":
        kw_hits = probe.get("keyword_hits", 0)
        vec_hits = probe.get("vector_hits", 0)
        top_vec = max((r.get("vec_score", 0.0) for r in results), default=0.0)
        agree = any(
            r.get("kw_rank", 99) <= 3 and r.get("vec_rank", 99) <= 3 for r in results
        )
        if vec_hits == 0:
            # keyword-only index — vector signals don't exist; judge (and
            # explain) from keyword strength alone
            if kw_hits > 0 and any(r.get("kw_rank") == 1 for r in results):
                used = "easy"
                reason = "用語がキーワード検索に直接ヒット（この索引はベクトル検索なし）"
            elif kw_hits == 0:
                used = "hard"
                reason = "キーワード一致が無いため（ベクトル検索なしの索引）、言い換えを生成して検索"
            else:
                used = "medium"
                reason = "キーワード検索のみの標準検索（この索引はベクトル検索なし）"
        elif results and (agree or (kw_hits > 0 and top_vec >= _STRONG_COS)):
            used = "easy"
            reason = "用語が資料に直接ヒットしたため、かんたん検索で十分と判断"
        elif kw_hits == 0 and top_vec < _WEAK_COS:
            used = "hard"
            reason = "キーワード一致が無く意味的な近さも弱いため、言い換えを生成して深く検索"
        else:
            used = "medium"
            reason = "標準のハイブリッド検索で十分な手応え"

    if used == "easy":
        sources = results[: _K["easy"]]
        reason = reason or "かんたん検索（1回のハイブリッド検索・上位のみ）"
    elif used == "medium":
        sources = results  # the probe already ran exactly this search
        reason = reason or "標準のハイブリッド検索"
    else:  # hard
        # dedup under the same normalization the index uses, so near-echoes
        # of the original query don't double-weight the RRF fusion (or spend
        # extra embedding calls)
        def _key(s: str) -> str:
            return " ".join(normalize(s).split())

        expansions: List[str] = []
        if expander is not None:
            try:
                seen = {_key(query)}
                for q in expander(query):
                    q = (q or "").strip()
                    k = _key(q)
                    if q and k and k not in seen:
                        seen.add(k)
                        expansions.append(q)
                    if len(expansions) == 3:
                        break
            except Exception:
                expansions = []
        queries = [query] + expansions
        if expansions:
            rank_lists = [
                index.search(q, mode="hybrid", k=12, include_text=True)["results"]
                for q in queries
            ]
            sources = _fuse(rank_lists, _K["hard"])
            reason = (reason + "。" if reason else "") + (
                f"{len(expansions)}件の言い換えでも検索し、順位を融合"
            )
        else:
            # no LLM available (demo model / no key / API error) — widen k instead
            sources = index.search(query, mode="hybrid", k=_K["hard"], include_text=True)[
                "results"
            ]
            reason = (reason + "。" if reason else "") + (
                "言い換え生成が使えないため、取得件数を広げて検索"
            )

    return {
        "sources": sources,
        "effort_requested": requested,
        "effort_used": used,
        "reason": reason,
        "queries": queries,
    }
