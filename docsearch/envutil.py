"""Tiny .env loader (avoids a python-dotenv dependency).

Looks for a .env next to the project root (doc-search/.env) and in the
current directory; existing environment variables always win.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_dotenv() -> None:
    for candidate in (PROJECT_ROOT / ".env", Path.cwd() / ".env"):
        if not candidate.is_file():
            continue
        for line in candidate.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip("'\"")
            # skip blank template values; a filled value only yields to a
            # non-empty real environment variable
            if value and not os.environ.get(key):
                os.environ[key] = value
