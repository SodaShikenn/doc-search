# build from the repo root:  docker compose up --build   (compose file at root)
# or: docker build -f deploy/Dockerfile -t docsearch .
#
# Two variants via build arg:
#   WITH_LOCAL_ML=0 (default)  slim ~300MB — vectors need VOYAGE_API_KEY
#                              (embedder auto/voyage/hash)
#   WITH_LOCAL_ML=1            +torch/sentence-transformers (~2-3GB) — fully
#                              offline vectors (embedder e5 / e5-large / bge-m3)
FROM python:3.12-slim

WORKDIR /app

ARG WITH_LOCAL_ML=0

COPY requirements.txt requirements-local.txt ./
RUN pip install --no-cache-dir -r requirements.txt && \
    if [ "$WITH_LOCAL_ML" = "1" ]; then \
        pip install --no-cache-dir -r requirements-local.txt; \
    fi

COPY docsearch/ docsearch/
COPY webui/ webui/
COPY sample_docs/ sample_docs/
COPY deploy/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV DOCSEARCH_DOCS=/docs \
    DOCSEARCH_INDEX=/data/index \
    HF_HOME=/data/hf-cache

EXPOSE 8765
VOLUME /data

ENTRYPOINT ["/entrypoint.sh"]
