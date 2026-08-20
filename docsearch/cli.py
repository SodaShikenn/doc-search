"""CLI: index / search / serve."""

from __future__ import annotations

import argparse
import json

from .envutil import load_dotenv


def main(argv=None) -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(
        prog="docsearch",
        description="Hybrid keyword + vector search over a documentation repo.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("index", help="build the index from a docs directory")
    p.add_argument("docs_dir")
    p.add_argument("--index-dir", default="index")
    p.add_argument("--no-vector", action="store_true", help="keyword index only")
    p.add_argument(
        "--embedder",
        default="auto",
        choices=["auto", "voyage", "e5", "e5-large", "bge-m3", "hash"],
        help="voyage: cloud, zero local deps / e5, e5-large, bge-m3: local "
        "(requirements-local.txt) / auto: voyage if key set, else e5 if "
        "installed, else hash (degraded)",
    )
    p.add_argument(
        "--github-base",
        default=None,
        help="blob-URL base for doc links, e.g. https://github.com/o/r/blob/main/docs "
        "(default: auto-detect from the docs repo's git remote; env "
        "DOCSEARCH_GITHUB_BASE also overrides)",
    )

    p = sub.add_parser("search", help="search from the command line")
    p.add_argument("query")
    p.add_argument("--index-dir", default="index")
    p.add_argument("--mode", default="hybrid", choices=["hybrid", "keyword", "vector"])
    p.add_argument("-k", type=int, default=10)
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("serve", help="start the web UI")
    p.add_argument("--index-dir", default="index")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)

    args = ap.parse_args(argv)

    if args.cmd == "index":
        from .indexer import build_index

        build_index(
            args.docs_dir,
            args.index_dir,
            with_vectors=not args.no_vector,
            embedder_name=args.embedder,
            github_base=args.github_base,
        )

    elif args.cmd == "search":
        from .search import SearchIndex

        out = SearchIndex(args.index_dir).search(args.query, mode=args.mode, k=args.k)
        if args.json:
            print(json.dumps(out, ensure_ascii=False, indent=2))
            return
        for i, r in enumerate(out["results"], start=1):
            marks = []
            if "kw_rank" in r:
                marks.append(f"kw#{r['kw_rank']}")
            if "vec_rank" in r:
                marks.append(f"vec#{r['vec_rank']} ({r['vec_score']:.3f})")
            print(f"{i:2}. {r['path']}:{r['line']}  [{' '.join(marks)}]")
            if r["heading"]:
                print(f"    {r['heading']}")
            print(f"    {r['snippet'][:160]}")
            print()

    elif args.cmd == "serve":
        import uvicorn

        from .server import create_app

        uvicorn.run(create_app(args.index_dir), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
