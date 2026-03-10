# ds_dojo4 - Rotten Tomatoes Review Prediction

Kaggle から移行したベースライン用のローカル環境です。

## フォルダ構成

```
ds_dojo4/
├── data/                    # データ (train.csv, test.csv) ※ .gitignore で除外可
├── docs/                    # ドキュメント（PROJECT_MIND.md＝方針・結果・やること1本、01〜04 前処理/特徴量/分析/指標）
├── lib/                     # ライブラリ（パイプライン・分析・encodings・OpenAI embedding）
├── preprocess/              # 前処理（load_train_test）
├── feature_engineering/    # 特徴量（create_features）
├── config/                  # API キー等（openai_api_key.txt）
├── outputs/                 # 実行結果（submissions/, analysis/, embeddings/）
├── train_baseline.ipynb    # ★ メイン：ベースライン学習・CV・提出（ここだけルートに配置）
├── archive/                 # その他ノート・スクリプト（参照・再利用用）
│   ├── run_openai_embeddings_once.py      # 1回だけ embedding 取得
│   ├── run_openai_three_submissions.py    # 3パターン提出一括
│   ├── run_baseline_openai_submission.py  # ベースライン+embedding 1本
│   ├── top_solutions.py                   # 上位解法・add_openai_embedding_features
│   ├── train_baseline_top_solutions.ipynb # 8種類提出
│   ├── train_baseline_staged_submissions.ipynb # 段階的提出
│   └── その他（baseline_pipeline.py, prediction_analysis.py 等）
├── requirements.txt
└── README.md
```

## 環境のセットアップ（Kaggle 互換 .venv）

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m ipykernel install --prefix .venv --name ds_dojo4 --display-name "Python 3 (ds_dojo4 .venv)"
```

（すでに実行済みの場合は不要です。）

## メインの使い方

- **いま使うノートブックは `train_baseline.ipynb` だけ**（ルートに1本）。カーネルを「Python 3 (ds_dojo4 .venv)」に設定し、プロジェクトルートで開いて実行する。`lib` からパイプライン・予測分析を利用し、時系列CVで学習・提出用 CSV を出力する。

## archive/ のスクリプト・ノート（参照・再利用）

以下は **archive/** に格納。**プロジェクトルートで** 実行すればそのまま動く。

- **embedding を1回だけ取得**: `python archive/run_openai_embeddings_once.py`（要 `config/openai_api_key.txt`）
- **ベースライン+embedding で3パターン提出**: `python archive/run_openai_three_submissions.py`
- **ベースライン+embedding で1本だけ**: `python archive/run_baseline_openai_submission.py`
- **ノートブック**: `archive/train_baseline_top_solutions.ipynb`（8種類提出）、`archive/train_baseline_staged_submissions.ipynb`（段階的提出）。開くときはカーネルをプロジェクトルートで起動するか、先頭のパス追加セルを最初に実行する。

詳細な方針・結果は **docs/PROJECT_MIND.md** を参照。

データは `data/` フォルダを参照しています。
