"""FastAPI server: search API, streaming RAG chat API, and the static web UI.

The chat endpoint streams `data: {json}\n\n` lines with the same envelope as
LLM-RAG_KBQA ({status, message, data}); data.type is one of
sources | delta | done | error.
"""

from __future__ import annotations

import json
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import llm, retriever
from .envutil import load_dotenv
from .search import SearchIndex

WEBUI_DIR = Path(__file__).resolve().parent.parent / "webui"
RAG_TOP_K = 6


def _sse(data: dict) -> str:
    payload = {"status": 200, "message": "ok", "data": data}
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


def create_app(index_dir: str = "index") -> FastAPI:
    load_dotenv()
    index = SearchIndex(index_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if index.embeddings is not None:
            # warm the embedding backend so the first vector query is instant
            threading.Thread(target=index.ensure_embedder, daemon=True).start()
        yield

    app = FastAPI(title="docsearch", lifespan=lifespan)

    @app.get("/api/search")
    def api_search(
        q: str = Query(""),
        mode: str = Query("hybrid", pattern="^(hybrid|keyword|vector)$"),
        k: int = Query(10, ge=1, le=50),
    ):
        if mode == "vector" and index.embeddings is None:
            raise HTTPException(400, "index was built without vectors (--no-vector)")
        t0 = time.perf_counter()
        out = index.search(q, mode=mode, k=k)
        out["took_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return out

    @app.get("/api/stats")
    def api_stats():
        return {
            **index.meta,
            "vectors_loaded": index.embeddings is not None,
            "model_ready": index._embedder is not None,
        }

    @app.get("/api/models")
    def api_models():
        return {
            "models": llm.MODELS,
            "default": llm.DEFAULT_MODEL,
            "api_key_present": llm.api_key_present(),
        }

    @app.post("/api/chat")
    def api_chat(payload: dict = Body(...)):
        raw_messages = payload.get("messages") or []
        params = payload.get("params") or {}
        model = params.get("model") or llm.DEFAULT_MODEL
        effort = params.get("effort", "auto")
        use_rag = bool(params.get("use_rag", True))
        if effort not in retriever.EFFORTS:
            raise HTTPException(400, "bad effort (use auto/easy/medium/hard)")

        # strip UI-side fields, keep the last 20 turns
        messages = [
            {"role": m["role"], "content": str(m.get("content", ""))}
            for m in raw_messages
            if m.get("role") in ("user", "assistant") and m.get("content")
        ][-20:]
        if not messages or messages[-1]["role"] != "user":
            raise HTTPException(400, "last message must be from user")

        def gen():
            try:
                yield from _chat_events()
            except Exception as e:  # noqa: BLE001 — headers are already sent;
                # the only way to surface a failure is a terminal SSE event
                import traceback

                traceback.print_exc()
                yield _sse(
                    {"type": "error", "message": f"サーバーエラーが発生しました: {type(e).__name__}"}
                )

        def _chat_events():
            sources = []
            retrieval = None
            if use_rag:
                yield _sse({"type": "status", "message": "資料を検索しています…"})
                query = messages[-1]["content"][:500]
                retrieval = retriever.retrieve(
                    index,
                    query,
                    effort=effort,
                    expander=lambda q: llm.expand_queries(q, model),
                )
                sources = retrieval["sources"]

            yield _sse(
                {
                    "type": "sources",
                    "github_base": index.meta.get("github_base"),
                    "sources": [
                        {
                            "path": s["path"],
                            "line": s["line"],
                            "heading": s["heading"],
                            "snippet": s["snippet"],
                            "url": s.get("url"),
                            "kw_rank": s.get("kw_rank"),
                            "vec_rank": s.get("vec_rank"),
                        }
                        for s in sources
                    ],
                    "retrieval": retrieval
                    and {
                        "effort_requested": retrieval["effort_requested"],
                        "effort_used": retrieval["effort_used"],
                        "reason": retrieval["reason"],
                        "queries": retrieval["queries"],
                    },
                }
            )
            system = llm.build_system(sources)
            effort_used = retrieval["effort_used"] if retrieval else "medium"
            yield from (
                _sse(ev)
                for ev in llm.stream_chat(model, system, messages, sources, effort_used)
            )

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    app.mount("/", StaticFiles(directory=WEBUI_DIR, html=True), name="webui")
    return app
