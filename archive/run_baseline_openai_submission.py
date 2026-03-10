#!/usr/bin/env python3
"""
ベースライン38 + OpenAI embedding 特徴（PCA 16次元）で学習し、提出用 CSV を1本作成する。
先に archive/run_openai_embeddings_once.py で embedding を保存しておくこと。プロジェクトルートで python archive/run_baseline_openai_submission.py を実行。
"""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
_archive = Path(__file__).resolve().parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_archive))

import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from lib import get_baseline_data, BASELINE_FEATURES, BASELINE_LGB_PARAMS
from top_solutions import (
    add_openai_embedding_features,
    get_time_splits,
    run_time_series_cv,
    save_submission,
)


def main():
    print("データ読み込み...", flush=True)
    train, test = get_baseline_data()
    y = train["target"].values
    print(f"  train={len(train)}, test={len(test)}", flush=True)

    print("OpenAI embedding 特徴を追加（PCA 16次元）...", flush=True)
    openai_names = add_openai_embedding_features(train, test)
    if not openai_names:
        raise SystemExit("OpenAI embedding が見つかりません。先に run_openai_embeddings_once.py を実行してください。")
    features = BASELINE_FEATURES + openai_names
    print(f"  特徴量数: {len(features)} (ベースライン38 + {len(openai_names)})", flush=True)

    time_splits = get_time_splits(train)
    print("時系列 CV...", flush=True)
    mean_auc, std_auc, test_preds = run_time_series_cv(
        train, test, features, y, time_splits,
        lgb_params=BASELINE_LGB_PARAMS,
        verbose=True,
    )
    print(f"CV AUC: {mean_auc:.4f} ± {std_auc:.4f}", flush=True)

    print("全データで学習して予測...", flush=True)
    model = lgb.LGBMClassifier(**BASELINE_LGB_PARAMS)
    model.fit(train[features], y)
    pred = model.predict_proba(test[features])[:, 1]

    out_path = "outputs/submissions/submission_baseline38_openai.csv"
    save_submission(test, pred, out_path)
    print(f"保存しました: {out_path}", flush=True)


if __name__ == "__main__":
    main()
