# doc-search — ハイブリッド検索 ＋ RAG チャット for ドキュメントリポジトリ

社内ドキュメントリポジトリ（用語集・レビュー観点・設計ドキュメント）を対象にした、
**キーワード検索（BM25）× ベクトル検索（意味検索）** のハイブリッド検索エンジンと、
その上に載る **RAG チャット（Claude API・モデル選択・ストリーミング出力）**。

UI 設計は [SodaShikenn/LLM-RAG_KBQA](https://github.com/SodaShikenn/LLM-RAG_KBQA) を踏襲
（左サイドバー: モデル選択 / ナレッジ設定 / 会話履歴、右: チャット + Send / Cancel）。

4つの使い方:

1. **RAG チャット** (`/`) — モデルを選んで質問。検索→引用付き回答をストリーミング
2. **検索エクスプローラ** (`/search.html`) — インクリメンタル検索、KW/VEC/RRF スコア表示
3. **CLI** — `docsearch search "..."`
4. **MCPサーバー** — Claude Code のツールとして登録（Agentic RAG）

## セットアップ（軽量: ML 依存なし・~30MB）

```bash
cd doc-search
brew install uv        # 未導入の場合
uv venv --python 3.12 .venv
uv pip install -p .venv/bin/python -r requirements.txt
cp .env.example .env   # ANTHROPIC_API_KEY を記入（チャット用）
```

ローカル埋め込みモデル（e5 / bge-m3）を使う場合のみ、重い ML スタックを追加:

```bash
uv pip install -p .venv/bin/python -r requirements-local.txt
```

## 使い方

```bash
# 1) インデックス構築
.venv/bin/python -m docsearch index sample_docs                    # 自動選択
.venv/bin/python -m docsearch index /path/to/docs --embedder voyage  # クラウド埋め込み

# 2) サーバー起動 → http://127.0.0.1:8765
.venv/bin/python -m docsearch serve --port 8765

# 3) CLI検索
.venv/bin/python -m docsearch search "解約率" --mode vector
```

APIキーが無くてもモデル「**Demo（オフライン）**」でチャットUIの動作確認ができる。

## 埋め込みモデルの選択 (`--embedder`)

| name | 場所 | 重さ | 特徴 |
|---|---|---|---|
| `voyage` | クラウド | **ローカル依存ゼロ** | voyage-3.5。Anthropic 推奨の埋め込みパートナー。品質最高クラス。`VOYAGE_API_KEY` 必要。文書テキストが外部送信される点は要承認 |
| `e5` | ローカル | ~470MB + torch | multilingual-e5-small。完全ローカル、日英対応の無難な既定 |
| `e5-large` | ローカル | ~2.2GB + torch | e5 の高精度版 |
| `bge-m3` | ローカル | ~2.3GB + torch | ローカル最強クラスの多言語モデル。ただし「軽量」とは真逆で CPU 推論も遅い |
| `hash` | ローカル | 依存ゼロ | 字面ハッシュ（意味検索なし・縮退モード） |

**選び方**: 品質とセットアップの軽さを両立したいなら `voyage`（クラウド許可時）。
完全ローカル必須なら `e5`、精度を上げたければ `bge-m3`（重さを許容できる場合）。
BM25（字句一致）は常にローカルで動くため、埋め込みの役割は「言い換え」の吸収のみ —
モデル差が効くのはそこだけで、`bge-m3` の multi-vector/sparse 機能はこの構成では不要。

## チャット (RAG) の仕組み

```
質問 → 検索の深さ（effort）を解決（auto は確信度シグナルで自動判断）
     → 検索実行（hard は選択モデルがクエリを言い換え → 全変種を検索して RRF 融合）
     → system プロンプトに参照資料として注入（[n] path:line 付き）
     → Claude API へストリーミング要求（output_config.effort も連動）
     → data: {status|sources|delta|done|error} を SSE 配信
     → UI が逐次描画 + 「なぜこの検索をしたか」の説明 + 引用チップ。会話は localStorage
```

### 検索の深さ（effort）— hybrid/keyword/vector を隠す

利用者に IR 用語を選ばせない。選ぶのは「どれだけしっかり探すか」だけで、
実際に何をしたかは回答の下に日本語で表示される（例: 「おまかせ → しっかり —
キーワード一致が無く…言い換えを生成して深く検索」）。

| effort | 動作 | 使いどころ |
|---|---|---|
| おまかせ (auto) | 一度探ってから確信度で easy/medium/hard を自動選択 | 既定。迷ったらこれ |
| かんたん (easy) | ハイブリッド検索1回・上位4件。モデルの effort も low | 用語の直接検索。最速・最安 |
| ふつう (medium) | 標準のハイブリッド検索・6件 | 従来の既定動作 |
| しっかり (hard) | 選択モデルが言い換えを3件生成 → 全クエリで検索し RRF 融合・10件。モデル effort は high | 資料と言葉遣いが違う質問（例:「残業した分の給料」→ 割増賃金） |

auto の判断シグナル: キーワード一致の有無・ベクトル類似度の強さ・両検索の上位一致。
言い換え生成が使えない場合（Demoモデル・キー未設定）は hard を「件数拡大」に自動縮退。
生の検索モード（keyword/vector/hybrid）はエンジニア向けに `/search.html` と CLI に残している。

- モデル: Claude Opus 5（既定）/ Sonnet 5 / Haiku 4.5 / Demo（オフライン）
- Opus 5 はサーバーサイド refusal fallback を有効化（安全上の回答辞退時に
  同一リクエスト内で代替モデルに自動フォールバック）
- 生成 API は Anthropic 公式 SDK。キーは `.env` の `ANTHROPIC_API_KEY`

## Claude Code への組み込み（MCP / Agentic RAG）

`.mcp.json`（対象リポジトリまたはホーム）:

```json
{
  "mcpServers": {
    "docsearch": {
      "command": "/ABSOLUTE/PATH/doc-search/.venv/bin/python",
      "args": ["-m", "docsearch.mcp_server"],
      "env": { "DOCSEARCH_INDEX": "/ABSOLUTE/PATH/doc-search/index" }
    }
  }
}
```

ツール: `search_docs(query, mode, k)` / `docs_repo_info()`。
Claude Code 自身がクエリ立案→再検索→ファイル読解→引用回答まで行うため、
チャットUIとは別に、エディタ内での Agentic RAG が成立する。

## 引用の GitHub リンク

検索結果・引用チップ・回答中の `[path:line]` は、ドキュメントリポジトリの
GitHub 上の該当行への深いリンクになる（`blob/<インデックス時のSHA>/path#L<line>`
形式なので、リポジトリが進んでも行アンカーはずれない）。

- インデックス時に docs リポジトリの `git remote` から**自動検出**（GHE も可）
- 自動検出できない場合（Docker で docs をマウントした場合など）は
  `DOCSEARCH_GITHUB_BASE=https://github.com/o/r/blob/main/docs` を `.env` に設定
  （CLI では `--github-base`）

## 検索エンジンの設計ポイント

- **日本語キーワード検索**: CJK文字列をバイグラム展開して SQLite FTS5 に索引。
  クエリ側はバイグラムのフレーズ検索で隣接一致（形態素解析器なしで動く）
- **RRF融合**: BM25スコアとコサイン類似度はスケール非互換のため順位ベースで融合
- **チャンクにパンくず**: 見出し階層をチャンク先頭に付与（用語集は見出し=用語のため）

## 試すと面白いクエリ

| クエリ | 期待 |
|---|---|
| `消費税区分` | キーワードで用語集に直撃 |
| `解約率` | ベクトルが「チャーンレート」を発見（言い換え） |
| `仕訳の二重登録を防ぐ仕組みは?`（チャット） | 冪等性 / Idempotency-Key を引用して回答 |
| `テナント 漏えい` | セキュリティ観点のテナント分離 |

## デプロイ

### ローカル常駐（macOS / LaunchAgent）

```bash
bash deploy/install-launchd.sh    # ログイン時自動起動・クラッシュ時自動再起動
```

- ログ: `logs/docsearch.log` / `logs/docsearch.err.log`
- 停止・削除: `launchctl bootout gui/$(id -u)/com.sodashikenn.docsearch && rm ~/Library/LaunchAgents/com.sodashikenn.docsearch.plist`
- **macOS TCC 注意**: リポジトリが `~/Desktop` 等の保護フォルダ配下にあると、
  launchd 起動の python がファイルアクセスを拒否されて起動ループになることがある。
  その場合は「システム設定 > プライバシーとセキュリティ」で python にアクセス権を
  与えるか、リポジトリを保護外（例: `~/dev/`）へ移動する

### Docker（別マシンとの共有はこれが最短）

```bash
git clone https://github.com/SodaShikenn/doc-search.git && cd doc-search
cp .env.example .env               # ANTHROPIC_API_KEY を記入
docker compose up --build -d       # → http://127.0.0.1:8765
```

- Docker が無い Mac（Docker Desktop を使わない場合）:

  ```bash
  brew install colima docker docker-compose && colima start
  mkdir -p ~/.docker/cli-plugins && ln -sfn $(brew --prefix)/opt/docker-compose/bin/docker-compose ~/.docker/cli-plugins/docker-compose
  ```

- **完全ローカルのベクトル検索にする場合**（メモリに余裕がある M シリーズ Mac 推奨）:
  `.env` に `WITH_LOCAL_ML=1` と `DOCSEARCH_EMBEDDER=e5` を書いてから
  `docker compose up --build -d`（イメージ ~2-3GB、初回はモデルDLあり。
  埋め込みモデルの変更は起動時に検知され自動で再索引される）
- 既定のスリム版イメージ（~300MB）: ベクトル検索は `VOYAGE_API_KEY` があればクラウド、
  無ければ hash 縮退（キーワード検索は常にフル動作）
- 実ドキュメントは `docker-compose.yml` の `./sample_docs:/docs:ro` を差し替え、
  内容更新後の再索引は `DOCSEARCH_REINDEX=1 docker compose up -d`
- キーはホストの `.env` から注入（イメージには焼き込まれない）
- **認証は無い**。公開はローカルバインドのままリバースプロキシ（認証付き）or VPN 越しに

## Third-party

`webui/vendor/` は自己ホストした第三者ライブラリで、各自のライセンスに従う:
[marked](https://github.com/markedjs/marked) v13.0.2 (MIT)・
[DOMPurify](https://github.com/cure53/DOMPurify) 3.1.6 (Apache-2.0 OR MPL-2.0)。
それ以外は MIT（LICENSE 参照）。

## 制限と発展

- インデックスは全再構築のみ（差分更新は未実装）
- 会話履歴はブラウザの localStorage（サーバー永続化なし）
- 評価: 質問→正解ファイルの recall@k で hybrid vs 単体・埋め込みモデル間を比較すると良い
