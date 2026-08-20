"""CJK-aware text normalization and tokenization.

Japanese has no word boundaries, and SQLite's unicode61 tokenizer treats a
whole CJK run as a single token, which makes substring queries impossible.
Instead we pre-tokenize: latin/digit runs stay whole words, CJK runs are
expanded into character bigrams. At query time a CJK run becomes a *phrase*
of its bigrams, which preserves adjacency, so「税区分」matches inside
「消費税区分」but not scattered characters.
"""

from __future__ import annotations

import unicodedata
from typing import Iterator, List, Optional, Tuple

_CJK_SINGLE = {0x3005, 0x3006, 0x3007}  # 々 〆 〇


def _char_class(ch: str) -> str:
    o = ord(ch)
    if (
        0x3040 <= o <= 0x30FF   # hiragana + katakana
        or 0x31F0 <= o <= 0x31FF  # katakana phonetic extensions
        or 0x3400 <= o <= 0x9FFF  # CJK unified ideographs (+ext A)
        or 0xF900 <= o <= 0xFAFF  # CJK compatibility ideographs
        or 0x20000 <= o <= 0x2FA1F  # CJK ext B..F
        or o in _CJK_SINGLE
    ):
        return "cjk"
    if ch.isalnum() or ch == "_":
        return "word"
    return "other"


def normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).lower()


def segments(text: str) -> Iterator[Tuple[str, str]]:
    """Yield (kind, run) for maximal runs of 'word' or 'cjk' chars."""
    buf: List[str] = []
    kind = ""
    for ch in text:
        cls = _char_class(ch)
        if cls == "other":
            if buf:
                yield kind, "".join(buf)
                buf, kind = [], ""
            continue
        if cls != kind and buf:
            yield kind, "".join(buf)
            buf = []
        kind = cls
        buf.append(ch)
    if buf:
        yield kind, "".join(buf)


def _bigrams(run: str) -> List[str]:
    if len(run) == 1:
        return [run]
    return [run[i : i + 2] for i in range(len(run) - 1)]


def index_tokens(text: str) -> List[str]:
    """Tokens to store in the FTS index (space-joined by the caller)."""
    toks: List[str] = []
    for kind, run in segments(normalize(text)):
        if kind == "word":
            toks.append(run)
        else:
            toks.extend(_bigrams(run))
    return toks


def fts_query(text: str) -> Optional[str]:
    """Build an FTS5 MATCH expression from a raw user query.

    - latin word  ->  "word" (the last one gets a prefix star for
      search-as-you-type)
    - CJK run     ->  phrase of its bigrams: "ab bc cd"
    - single CJK char -> prefix query "x"* so it matches any bigram
      starting with that character
    Segments are ANDed.
    """
    parts: List[str] = []
    segs = list(segments(normalize(text)))
    for i, (kind, run) in enumerate(segs):
        run = run.replace('"', '""')
        if kind == "word":
            star = "*" if i == len(segs) - 1 else ""
            parts.append(f'"{run}"{star}')
        elif len(run) == 1:
            parts.append(f'"{run}"*')
        else:
            phrase = " ".join(_bigrams(run))
            parts.append(f'"{phrase}"')
    return " AND ".join(parts) if parts else None


def query_terms(text: str) -> List[str]:
    """Normalized query segments, used for snippet extraction/highlighting."""
    return [run for _, run in segments(normalize(text))]
