"""Dependency-free smoke tests: python tests/test_basic.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os  # noqa: E402

from docsearch.chunker import chunk_file  # noqa: E402
from docsearch.datasource import PLACEHOLDER, get_datasource  # noqa: E402
from docsearch.gitlink import doc_url, parse_remote  # noqa: E402
from docsearch.search import _snippet  # noqa: E402
from docsearch.textutil import fts_query, index_tokens, query_terms  # noqa: E402


def test_datasource(tmp_path: Path):
    # no config file -> the built-in placeholder
    ds = get_datasource(tmp_path / "missing.json")
    assert ds.docs_dir == PLACEHOLDER.docs_dir
    assert "placeholder" in ds.name
    # config file replaces the placeholder (how the data-owning machine plugs in)
    cfg = tmp_path / "ds.json"
    cfg.write_text(
        '{"name":"real","docs_dir":"/data/docs",'
        '"github_base":"https://ghe.corp/o/r/blob/main","embedder":"voyage"}',
        encoding="utf-8",
    )
    ds = get_datasource(cfg)
    assert (ds.docs_dir, ds.embedder) == ("/data/docs", "voyage")
    # env vars override individual fields; the rest survive
    os.environ["DOCSEARCH_DOCS"] = "/env/docs"
    try:
        ds = get_datasource(cfg)
        assert ds.docs_dir == "/env/docs"
        assert ds.github_base == "https://ghe.corp/o/r/blob/main"
    finally:
        del os.environ["DOCSEARCH_DOCS"]


def test_gitlink():
    assert parse_remote("git@github.com:o/r.git") == "https://github.com/o/r"
    assert parse_remote("https://github.com/o/r") == "https://github.com/o/r"
    assert parse_remote("https://user@ghe.corp.jp/team/docs.git") == "https://ghe.corp.jp/team/docs"
    assert parse_remote("ssh://git@ghe.corp.jp/team/docs.git") == "https://ghe.corp.jp/team/docs"
    assert parse_remote("/local/path") is None
    assert parse_remote("") is None
    base = "https://github.com/o/r/blob/abc123/docs"
    assert doc_url(base, "payroll/glossary.md", 35) == base + "/payroll/glossary.md#L35"
    assert doc_url(base, "日本語 名.md", 1) == base + "/%E6%97%A5%E6%9C%AC%E8%AA%9E%20%E5%90%8D.md#L1"
    assert doc_url(None, "a.md", 1) is None


def test_snippet_nfkc_offsets():
    # half-width katakana normalizes to different lengths — the snippet must
    # still cut at the right place in the ORIGINAL text
    text = ("あ" * 300) + "ﾃﾞｰﾀﾍﾞｰｽの設計" + ("い" * 300)
    snip = _snippet(text, ["データベース"])
    assert "ﾃﾞｰﾀﾍﾞｰｽ" in snip
    assert _snippet("短いテキスト", ["みつからない"]).startswith("短い")


def test_tokenize():
    assert index_tokens("消費税区分") == ["消費", "費税", "税区", "区分"]
    assert index_tokens("N+1 クエリ") == ["n", "1", "クエ", "エリ"]
    assert index_tokens("Idempotency-Key") == ["idempotency", "key"]
    assert index_tokens("税") == ["税"]


def test_fts_query():
    assert fts_query("税区分") == '"税区 区分"'
    assert fts_query("税") == '"税"*'
    assert fts_query("review 観点") == '"review" AND "観点"'  # trailing word gets *? no: 観点 is cjk
    assert fts_query("secur") == '"secur"*'
    assert fts_query("!!!") is None
    assert query_terms("テナント 漏えい") == ["テナント", "漏えい"]


def test_chunker(tmp_path: Path):
    md = tmp_path / "t.md"
    md.write_text(
        "# 用語集\n\n前文。\n\n## 仕訳\n\n取引を記録すること。\n\n"
        "```\n# not a heading\n```\n\n## 勘定科目\n\nラベル。\n",
        encoding="utf-8",
    )
    chunks = chunk_file(md, "t.md")
    heads = [c.heading for c in chunks]
    assert "用語集" in heads
    assert "用語集 > 仕訳" in heads
    assert "用語集 > 勘定科目" in heads
    ledger = next(c for c in chunks if c.heading.endswith("仕訳"))
    assert "# not a heading" in ledger.text  # fenced line stayed in body
    assert ledger.start_line == 7


if __name__ == "__main__":
    import tempfile

    test_tokenize()
    test_fts_query()
    test_gitlink()
    with tempfile.TemporaryDirectory() as d:
        test_chunker(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_datasource(Path(d))
    test_snippet_nfkc_offsets()
    print("all tests passed")
