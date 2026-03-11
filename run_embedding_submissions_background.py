#!/usr/bin/env python3
"""
Embedding 次元削減の残り手法で提出ファイルを 1 本ずつ作成し、outputs/submissions/ に出来次第保存する。
裏で回す用: nohup で実行するとログに進捗が出る。

使い方:
  # 残り 6 手法のみ（既に存在する CSV はスキップ）
  nohup python run_embedding_submissions_background.py --skip-existing > outputs/embedding_submissions.log 2>&1 &

  # 全 8 手法を上書きで作成
  nohup python run_embedding_submissions_background.py > outputs/embedding_submissions.log 2>&1 &

  # ログ確認
  tail -f outputs/embedding_submissions.log

注意: パソコンの蓋を閉じるとスリープで処理が止まります。裏で回す間は蓋を開けたままか、
スリープを無効にするか、別マシン（リモートサーバ等）で実行してください。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import lightgbm as lgb

from lib import get_baseline_data, BASELINE_FEATURES, BASELINE_LGB_PARAMS
from lib.embedding_reduction import REDUCTION_METHODS, fit_transform_embedding
from lib.submission import save_submission, verify_submission

# ノートブックと同じ設定（最良: movie_title_info + 16 次元）
EMBEDDING_NAME = "movie_title_info"
N_COMPONENTS = 16
OUTPUT_DIR = ROOT / "outputs"
EMBEDDINGS_DIR = OUTPUT_DIR / "embeddings"
SUBMISSIONS_DIR = OUTPUT_DIR / "submissions"

EMBEDDING_CONFIGS = {
    "movie_info": {"path": EMBEDDINGS_DIR / "movie_info_embeddings.pkl", "loader": "movie_info"},
    "movie_info_large": {"path": EMBEDDINGS_DIR / "movie_info_embeddings_large.pkl", "loader": "movie_info"},
    "movie_title_info": {"path": EMBEDDINGS_DIR / "movie_title_info_embeddings.pkl", "loader": "title_info"},
    "movie_title_info_large": {"path": EMBEDDINGS_DIR / "movie_title_info_embeddings_large.pkl", "loader": "title_info"},
}


def load_embeddings(name: str) -> pd.DataFrame:
    if name not in EMBEDDING_CONFIGS:
        raise ValueError(f"不明な embedding: {name}")
    config = EMBEDDING_CONFIGS[name]
    path = Path(config["path"])
    if not path.exists():
        path = path.with_suffix(".parquet")
    if not path.exists():
        raise FileNotFoundError(f"embedding がありません: {config['path']}")
    if config["loader"] == "movie_info":
        from lib.openai_embeddings import load_movie_info_embeddings
        return load_movie_info_embeddings(path=path)
    from lib.openai_embeddings import load_movie_title_info_embeddings
    return load_movie_title_info_embeddings(path=path)


def _merge_embeddings(df: pd.DataFrame, emb_df: pd.DataFrame) -> np.ndarray:
    emb_cols = [c for c in emb_df.columns if c != "rotten_tomatoes_link"]
    merged = df[["rotten_tomatoes_link"]].merge(
        emb_df[["rotten_tomatoes_link"] + emb_cols], on="rotten_tomatoes_link", how="left"
    )
    return merged[emb_cols].fillna(0).astype(np.float32).values


def main():
    parser = argparse.ArgumentParser(description="Embedding 提出ファイルを次元削減手法ごとに作成（裏実行用）")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="既に提出 CSV が存在する手法はスキップ（残りだけ回す）",
    )
    args = parser.parse_args()

    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
    print(f"提出先: {SUBMISSIONS_DIR}", flush=True)
    print(f"embedding: {EMBEDDING_NAME}, 次元: {N_COMPONENTS}", flush=True)
    print(f"手法: {REDUCTION_METHODS}", flush=True)
    if args.skip_existing:
        print("--skip-existing: 既存ファイルはスキップします", flush=True)
    print("", flush=True)

    train, test = get_baseline_data()
    y = train["target"].values
    base_features = list(BASELINE_FEATURES)
    emb_df = load_embeddings(EMBEDDING_NAME)
    E_train = _merge_embeddings(train, emb_df)
    E_test = _merge_embeddings(test, emb_df)
    safe_emb = EMBEDDING_NAME.replace(" ", "_")

    saved = []
    for method in REDUCTION_METHODS:
        fname = f"submission_embedding_{safe_emb}_{method}{N_COMPONENTS}.csv"
        out_path = SUBMISSIONS_DIR / fname
        if args.skip_existing and out_path.exists():
            print(f"  [{method}] スキップ（既存）: {fname}", flush=True)
            saved.append(fname)
            continue
        print(f"  [{method}] 次元削減・学習中（数分かかることがあります）...", flush=True)
        try:
            train_reduced, test_reduced, prefix = fit_transform_embedding(
                E_train, E_test, method=method, n_components=N_COMPONENTS, random_state=42
            )
        except Exception as e:
            print(f"  [{method}] スキップ: {e}", flush=True)
            continue
        print(f"  [{method}] 学習・予測中...", flush=True)
        n_comp = train_reduced.shape[1]
        red_names = [f"{prefix}_{i}" for i in range(n_comp)]
        X_train = pd.concat([
            train[base_features].reset_index(drop=True),
            pd.DataFrame(train_reduced, columns=red_names),
        ], axis=1)
        X_test = pd.concat([
            test[base_features].reset_index(drop=True),
            pd.DataFrame(test_reduced, columns=red_names),
        ], axis=1)
        feats = base_features + red_names
        model = lgb.LGBMClassifier(**BASELINE_LGB_PARAMS)
        model.fit(X_train[feats], y)
        pred = model.predict_proba(X_test[feats])[:, 1]
        save_submission(test, pred, out_path, sanitize=True)
        v = verify_submission(out_path, test)
        status = "OK" if v["ok"] else v["message"]
        print(f"  [{method}] → {fname}  ({status})", flush=True)
        saved.append(fname)

    print("", flush=True)
    print(f"保存先: {SUBMISSIONS_DIR}", flush=True)
    print(f"作成した提出ファイル: {len(saved)} 件", flush=True)


if __name__ == "__main__":
    main()
