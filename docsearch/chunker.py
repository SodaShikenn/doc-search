"""Markdown-aware chunking.

Documents are split at ATX headings (respecting code fences), and each
section keeps its full heading breadcrumb (H1 > H2 > ...). Long sections are
packed into chunks of ~MAX_CHARS on paragraph boundaries so a chunk never
cuts a sentence in half. The breadcrumb is prepended to the text used for
indexing/embedding — a glossary entry's term usually lives in the heading.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

MAX_CHARS = 900

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")


@dataclass
class Chunk:
    path: str          # relative path within the docs repo
    heading: str       # breadcrumb, e.g. "用語集 > 仕訳"
    start_line: int    # 1-based line the chunk starts at
    text: str          # chunk body (original markdown)

    @property
    def index_text(self) -> str:
        """Text fed to both the FTS tokenizer and the embedder."""
        return f"{self.heading}\n{self.text}" if self.heading else self.text


def _sections(lines: List[str]) -> List[Tuple[List[str], int, List[Tuple[int, str]]]]:
    """Split into (breadcrumb, start_line, [(line_no, paragraph)]) sections."""
    sections = []
    crumb: List[str] = []
    para_lines: List[str] = []
    para_start = 1
    paras: List[Tuple[int, str]] = []
    sec_start = 1
    in_fence = False

    def flush_para(next_line: int) -> None:
        nonlocal para_lines
        text = "\n".join(para_lines).strip()
        if text:
            paras.append((para_start, text))
        para_lines = []

    def flush_section(next_start: int, new_crumb: List[str]) -> None:
        nonlocal paras, sec_start, crumb
        flush_para(next_start)
        if paras:
            sections.append((list(crumb), sec_start, paras))
        paras = []
        sec_start = next_start
        crumb = new_crumb

    for no, line in enumerate(lines, start=1):
        if _FENCE.match(line):
            in_fence = not in_fence
        m = None if in_fence else _HEADING.match(line)
        if m:
            level, title = len(m.group(1)), m.group(2)
            new_crumb = crumb[: level - 1] + [title]
            flush_section(no, new_crumb)
            continue
        if line.strip():
            if not para_lines:
                para_start = no
            para_lines.append(line)
        else:
            flush_para(no)
    flush_section(len(lines) + 1, [])
    return sections


def _pack(paras: List[Tuple[int, str]], max_chars: int) -> List[Tuple[int, str]]:
    """Pack paragraphs into chunks of at most ~max_chars."""
    chunks: List[Tuple[int, str]] = []
    buf: List[str] = []
    buf_start = 0
    size = 0
    for line_no, para in paras:
        # hard-split a pathological single paragraph
        while len(para) > max_chars * 2:
            head, para = para[:max_chars], para[max_chars:]
            if buf:
                chunks.append((buf_start, "\n\n".join(buf)))
                buf, size = [], 0
            chunks.append((line_no, head))
        if buf and size + len(para) > max_chars:
            chunks.append((buf_start, "\n\n".join(buf)))
            buf, size = [], 0
        if not buf:
            buf_start = line_no
        buf.append(para)
        size += len(para)
    if buf:
        chunks.append((buf_start, "\n\n".join(buf)))
    return chunks


def chunk_file(path: Path, rel_path: str, max_chars: int = MAX_CHARS) -> List[Chunk]:
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []
    lines = text.splitlines()
    result: List[Chunk] = []
    if path.suffix.lower() in (".md", ".markdown", ".mdx"):
        for crumb, _sec_start, paras in _sections(lines):
            heading = " > ".join(crumb)
            for start, body in _pack(paras, max_chars):
                result.append(Chunk(rel_path, heading, start, body))
    else:  # plain text
        paras = []
        start, buf = 1, []
        for no, line in enumerate(lines, start=1):
            if line.strip():
                if not buf:
                    start = no
                buf.append(line)
            elif buf:
                paras.append((start, "\n".join(buf)))
                buf = []
        if buf:
            paras.append((start, "\n".join(buf)))
        for s, body in _pack(paras, max_chars):
            result.append(Chunk(rel_path, Path(rel_path).stem, s, body))
    return result
