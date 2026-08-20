#!/bin/sh
# Build the index on first boot (or when DOCSEARCH_REINDEX=1), then serve.
set -e

DOCS_DIR="${DOCSEARCH_DOCS:-/docs}"
INDEX_DIR="${DOCSEARCH_INDEX:-/data/index}"
EMBEDDER="${DOCSEARCH_EMBEDDER:-auto}"

# reject configs that can only crash-loop
case "$EMBEDDER" in
    auto|voyage|hash) ;;
    e5|e5-large|bge-m3)
        if ! python -c "import sentence_transformers" 2>/dev/null; then
            echo "error: DOCSEARCH_EMBEDDER=$EMBEDDER needs local ML deps not in this image." >&2
            echo "Rebuild with: WITH_LOCAL_ML=1 docker compose up --build" >&2
            exit 64
        fi ;;
    *)
        echo "error: unknown DOCSEARCH_EMBEDDER=$EMBEDDER" >&2
        echo "use auto, voyage, hash — or e5 / e5-large / bge-m3 with WITH_LOCAL_ML=1" >&2
        exit 64 ;;
esac
if [ "$EMBEDDER" = "voyage" ] && [ -z "$VOYAGE_API_KEY" ]; then
    echo "error: DOCSEARCH_EMBEDDER=voyage but VOYAGE_API_KEY is empty" >&2
    exit 64
fi

# bare `docker run` without a /docs mount: fall back to the baked sample docs
if [ ! -d "$DOCS_DIR" ] || [ -z "$(ls -A "$DOCS_DIR" 2>/dev/null)" ]; then
    echo "warning: $DOCS_DIR is empty — using baked-in sample_docs" >&2
    DOCS_DIR=/app/sample_docs
fi

REBUILD="$DOCSEARCH_REINDEX"
if [ -f "$INDEX_DIR/meta.json" ] && [ "$EMBEDDER" != "auto" ]; then
    # the index remembers its embedder on the persistent volume; a changed
    # DOCSEARCH_EMBEDDER would otherwise be silently ignored
    CURRENT=$(python -c "import json;print(json.load(open('$INDEX_DIR/meta.json')).get('embedder') or '')")
    if [ "$CURRENT" != "$EMBEDDER" ]; then
        echo "embedder changed ($CURRENT -> $EMBEDDER) — rebuilding index" >&2
        REBUILD=1
    fi
fi

if [ ! -f "$INDEX_DIR/meta.json" ] || [ "$REBUILD" = "1" ]; then
    echo "building index from $DOCS_DIR (embedder: $EMBEDDER) ..."
    python -m docsearch index "$DOCS_DIR" \
        --index-dir "$INDEX_DIR" \
        --embedder "$EMBEDDER"
fi

exec python -m docsearch serve --host 0.0.0.0 --port 8765 --index-dir "$INDEX_DIR"
