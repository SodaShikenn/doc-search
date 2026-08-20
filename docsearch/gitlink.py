"""Resolve GitHub (or GHE) deep links for indexed documents.

At index time we detect the docs repo's remote and record a blob-URL base in
meta.json; at query time every result gets `url` = base/path#Lline. The ref
is the commit SHA at index time, so line anchors always match the indexed
snapshot even after the repo moves on.

Override with DOCSEARCH_GITHUB_BASE (or --github-base) when auto-detection
can't work — e.g. inside Docker where the docs are mounted without .git —
using any prefix ending at the directory that maps to the indexed docs_dir,
e.g. https://github.com/owner/repo/blob/main/docs
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Optional
from urllib.parse import quote

_REMOTE_RES = [
    re.compile(r"^git@(?P<host>[^:/]+):(?P<path>.+?)(?:\.git)?/?$"),
    re.compile(r"^ssh://git@(?P<host>[^/]+)/(?P<path>.+?)(?:\.git)?/?$"),
    re.compile(r"^https?://(?:[^@/]+@)?(?P<host>[^/]+)/(?P<path>.+?)(?:\.git)?/?$"),
]


def parse_remote(url: str) -> Optional[str]:
    """'git@github.com:o/r.git' → 'https://github.com/o/r' (None if unknown)."""
    url = (url or "").strip()
    for rx in _REMOTE_RES:
        m = rx.match(url)
        if m:
            path = m.group("path").strip("/")
            if "/" in path:
                return f"https://{m.group('host')}/{path}"
    return None


def _git(args: list, cwd) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def detect_github_base(docs_dir) -> Optional[str]:
    """Env override, else derive a blob-URL base from the docs repo's remote.

    Returns e.g. https://github.com/o/r/blob/<sha>[/subdir], or None.
    """
    override = os.environ.get("DOCSEARCH_GITHUB_BASE")
    if override:
        return override.rstrip("/")
    docs_dir = Path(docs_dir).resolve()
    top = _git(["rev-parse", "--show-toplevel"], docs_dir)
    repo_url = parse_remote(_git(["config", "--get", "remote.origin.url"], docs_dir) or "")
    sha = _git(["rev-parse", "HEAD"], docs_dir)
    if not (top and repo_url and sha):
        return None
    prefix = docs_dir.relative_to(Path(top)).as_posix()
    base = f"{repo_url}/blob/{sha}"
    return f"{base}/{prefix}" if prefix not in ("", ".") else base


def doc_url(base: Optional[str], path: str, line: Optional[int] = None) -> Optional[str]:
    if not base:
        return None
    url = f"{base}/{quote(str(path))}"
    return f"{url}#L{int(line)}" if line else url
