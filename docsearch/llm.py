"""Claude API layer: model registry + streaming RAG chat.

Uses the official Anthropic SDK. Credentials resolve from the environment
(ANTHROPIC_API_KEY, loaded from doc-search/.env if present). For Claude
Opus 5 the server-side refusal-fallback is enabled by default, so a safety
decline re-runs the request on a fallback model within the same call.

A "demo" pseudo-model streams a canned answer offline — same event shape —
so the UI can be demonstrated without any API key (mirrors the /chat/test
endpoint in LLM-RAG_KBQA).
"""

from __future__ import annotations

import os
import re
import time
from typing import Dict, Iterator, List

from .envutil import load_dotenv

MODELS = [
    {"id": "claude-opus-5", "label": "Claude Opus 5"},
    {"id": "claude-sonnet-5", "label": "Claude Sonnet 5"},
    {"id": "claude-haiku-4-5", "label": "Claude Haiku 4.5"},
    {"id": "demo", "label": "Demo（オフライン・API キー不要）"},
]
DEFAULT_MODEL = "claude-opus-5"

SYSTEM_BASE = """あなたは社内ドキュメント検索アシスタントです。用語集・レビュー観点・設計ドキュメントを検索した結果（参照資料）に基づいて、エンジニアの質問に日本語で簡潔に答えてください。

ルール:
- 回答は参照資料に基づくこと。根拠となる資料を文末に [path:line] の形式で引用する
- 参照資料に答えがない場合は、その旨を明言した上で一般知識と分けて答える
- 用語は資料内の表記（用語集の見出し）に合わせる"""


def api_key_present() -> bool:
    load_dotenv()
    return bool(
        os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    )


# retrieval effort -> Claude output_config.effort (only Opus 5 / Sonnet 5
# accept the parameter; Haiku 4.5 rejects it)
_EFFORT_TO_API = {"easy": "low", "medium": "medium", "hard": "high"}
_EFFORT_CAPABLE = {"claude-opus-5", "claude-sonnet-5"}


def expand_queries(query: str, model: str, n: int = 3) -> List[str]:
    """Generate query reformulations for hard-effort retrieval.

    Uses the user-selected chat model. Returns [] when no key / demo model /
    API error — the retriever degrades gracefully.
    """
    if model == "demo" or not api_key_present():
        return []
    import anthropic

    prompt = (
        "社内ドキュメント検索のクエリ言い換えを生成してください。"
        f"元のクエリ:「{query}」\n\n"
        f"検索でヒットしやすい言い換えを{n}個、1行に1個だけ出力してください。"
        "日本語の同義語・用語集にありそうな正式な用語・英語の専門用語を混ぜること。"
        "番号・記号・説明は不要です。"
    )
    kwargs: Dict = dict(
        model=model,
        # opus-5/sonnet-5 think adaptively by default and thinking counts
        # against max_tokens — leave generous headroom (unused isn't billed)
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )
    if model in _EFFORT_CAPABLE:
        kwargs["output_config"] = {"effort": "low"}  # mechanical task
    try:
        resp = anthropic.Anthropic().messages.create(**kwargs)
        text = next((b.text for b in resp.content if b.type == "text"), "")
        # remove only a leading list marker — a charset strip would mangle
        # terms that start/end with digits (2要素認証, S3, ISO27001)
        lines = [
            re.sub(r"^\s*(?:\d{1,2}[.)．）、]|[・\-*•])\s*", "", ln).strip()
            for ln in text.splitlines()
        ]
        lines = [ln for ln in lines if ln]
        if resp.stop_reason == "max_tokens" and lines:
            lines = lines[:-1]  # last line may be cut mid-word
        return lines[:n]
    except anthropic.APIError:
        return []


def build_system(sources: List[Dict]) -> str:
    if not sources:
        return (
            SYSTEM_BASE
            + "\n\n（今回はドキュメント検索が無効か、該当する資料が見つかりませんでした。"
            "その前提を明示して答えてください。）"
        )
    blocks = []
    for i, s in enumerate(sources, start=1):
        head = f" | {s['heading']}" if s.get("heading") else ""
        blocks.append(f"[{i}] {s['path']}:{s['line']}{head}\n{s['text']}")
    return SYSTEM_BASE + "\n\n# 参照資料\n\n" + "\n\n---\n\n".join(blocks)


def _demo_stream(sources: List[Dict]) -> Iterator[Dict]:
    cites = "、".join(f"`{s['path']}:{s['line']}`" for s in sources[:3]) or "（なし）"
    text = (
        "これはオフラインのデモ応答です（APIキー未設定でもUIの動作確認ができます）。"
        f"\n\n検索エンジンは今回の質問に対して {len(sources)} 件のチャンクを取得しました。"
        f"上位の参照元: {cites}\n\n"
        "実際のモデル（Claude Opus 5 など）を選ぶと、この参照資料に基づいた回答が"
        "ストリーミングされます。`.env` に `ANTHROPIC_API_KEY` を設定してください。"
    )
    for i in range(0, len(text), 6):
        time.sleep(0.02)
        yield {"type": "delta", "content": text[i : i + 6]}
    yield {"type": "done", "model": "demo", "output_tokens": 0, "stop_reason": "end_turn"}


def stream_chat(
    model: str,
    system: str,
    messages: List[Dict],
    sources: List[Dict],
    effort: str = "medium",
) -> Iterator[Dict]:
    """Yield {'type': 'delta'|'done'|'error', ...} events."""
    if model == "demo":
        yield from _demo_stream(sources)
        return

    load_dotenv()
    import anthropic

    if not api_key_present():
        yield {
            "type": "error",
            "message": "ANTHROPIC_API_KEY が未設定です。doc-search/.env に "
            "ANTHROPIC_API_KEY=sk-ant-... を書いてサーバーを再起動してください"
            "（オフラインで試すにはモデル『Demo』を選択）。",
        }
        return

    kwargs: Dict = dict(model=model, max_tokens=16000, system=system, messages=messages)
    if model in _EFFORT_CAPABLE:
        kwargs["output_config"] = {"effort": _EFFORT_TO_API.get(effort, "medium")}
    if model == "claude-opus-5":
        # server-side refusal fallback: a policy decline re-runs on a fallback
        # model inside the same request instead of returning an empty answer
        kwargs["betas"] = ["server-side-fallback-2026-07-01"]
        kwargs["fallbacks"] = "default"

    client = anthropic.Anthropic()
    try:
        with client.beta.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                yield {"type": "delta", "content": text}
            final = stream.get_final_message()
        if final.stop_reason == "refusal":
            detail = ""
            if getattr(final, "stop_details", None) and final.stop_details.explanation:
                detail = f"（{final.stop_details.explanation}）"
            yield {"type": "delta", "content": f"\n\n※ モデルが回答を辞退しました{detail}"}
        yield {
            "type": "done",
            "model": final.model,
            "output_tokens": final.usage.output_tokens,
            "stop_reason": final.stop_reason,
        }
    except anthropic.AuthenticationError:
        yield {"type": "error", "message": "APIキーが無効です（.env の ANTHROPIC_API_KEY を確認）。"}
    except anthropic.NotFoundError:
        yield {"type": "error", "message": f"モデル {model} が見つかりません。"}
    except anthropic.RateLimitError:
        yield {"type": "error", "message": "レート制限に達しました。少し待って再試行してください。"}
    except anthropic.APIStatusError as e:
        yield {"type": "error", "message": f"APIエラー ({e.status_code}): {e.message}"}
    except anthropic.APIConnectionError:
        yield {"type": "error", "message": "Anthropic API に接続できません（ネットワークを確認）。"}
    except anthropic.APIError as e:  # e.g. APIResponseValidationError
        yield {"type": "error", "message": f"APIエラー: {type(e).__name__}"}
