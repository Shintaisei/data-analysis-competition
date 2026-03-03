# ds_dojo4 - Rotten Tomatoes Review Prediction

Kaggle から移行したベースライン用のローカル環境です。

## フォルダ構成

```
ds_dojo4/
├── data/                    # データ (train.csv, test.csv) ※ .gitignore で除外可
├── docs/                    # ドキュメント（01 前処理, 02 特徴量, 03 分析, 04 評価指標）
├── outputs/                 # 実行結果（ノートから自動出力）
│   ├── submissions/         # 提出用 CSV (submission.csv, submission_*.csv)
│   ├── analysis/            # 予測分析 (train_with_predictions.csv, prediction_summary_by_*.csv)
│   └── experiments/        # 実験結果 (metric_driven_*.csv)
├── preprocess/              # 前処理（データ読み込み）
│   └── preprocess.py        # get_data_dir(), load_train_test()
├── feature_engineering/     # 特徴量
│   └── features.py         # create_features()
├── lib/                     # ノートブック用ライブラリ（パイプライン・分析・エンコーディング・テキストベクトル）
│   ├── pipeline.py          # get_baseline_data, BASELINE_FEATURES
│   ├── analysis.py          # add_prediction_analysis, run_full_analysis 等
│   ├── encodings.py         # 時系列 TE, movie_info_meta 等
│   └── text_vectors.py     # build_vectors, get_available_configs 等
├── train_baseline.ipynb    # ベースライン学習・CV・提出・予測分析
├── train_metric_driven_experiments.ipynb  # 改善実験（評価指標拡張・複数 config 比較）
├── 終わった実験環境/        # 過去ノート（参照用）
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

## ノートブックの実行

1. カーネルを **「Python 3 (ds_dojo4 .venv)」** または **「.venv」** に設定
2. 実行するノートブックを開く（プロジェクトルートでカーネルを起動すること）
   - **`train_baseline.ipynb`**: `lib` からパイプライン・予測分析を利用。時系列CVで学習し提出用 CSV を出力。**いじらず基準として残す。**
   - **`train_metric_driven_experiments.ipynb`**: 評価指標を拡張した改善実験。複数 config を比較し、ベスト設定の提出用 CSV を出力。
   - **`train_lgb_tuning.ipynb`**: LightGBM のハイパーパラメータを Optuna で探索。特徴量はベースラインのまま、目先のスコア上げ用。
   - **`終わった実験環境/`**: 過去の実験ノート（前処理比較・特徴量・テキストベクトル等）は参照用。

データは `data/` フォルダを参照しています。
