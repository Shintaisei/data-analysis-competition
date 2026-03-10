#!/usr/bin/env python3
"""
OpenAI embedding 特徴の「使い方」を3パターンに厳選し、提出用 CSV を3本作成する。

パターン1: ベースライン38 + OpenAI 16次元 … シンプルにテキスト意味を追加
パターン2: ベースライン38 + OpenAI 8次元   … 次元を絞って過学習を抑制
パターン3: v2（OOF TE+カウント）+ OpenAI 16次元 … 協調的な特徴とテキスト意味の組み合わせ

先に archive/run_openai_embeddings_once.py で outputs/embeddings/ に embedding を保存しておくこと。プロジェクトルートで python archive/run_openai_three_submissions.py を実行。
"""
import sys
from pathlib import Path

# archive/ から実行しても lib と top_solutions を参照できるようにする
_root = Path(__file__).resolve().parent.parent
_archive = Path(__file__).resolve().parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_archive))

import lightgbm as lgb

from lib import get_baseline_data, BASELINE_FEATURES, BASELINE_LGB_PARAMS
from top_solutions import (
    V2_EXTRA_NAMES,
    add_openai_embedding_features,
    add_v2_features,
    get_time_splits,
    run_time_series_cv,
    save_submission,
)


def run_one(train, test, y, features, name, out_suffix):
    """1パターン分の時系列CV・全データ学習・提出保存。"""
    time_splits = get_time_splits(train)
    mean_auc, std_auc, _ = run_time_series_cv(
        train, test, features, y, time_splits,
        lgb_params=BASELINE_LGB_PARAMS,
        verbose=False,
    )
    model = lgb.LGBMClassifier(**BASELINE_LGB_PARAMS)
    model.fit(train[features], y)
    pred = model.predict_proba(test[features])[:, 1]
    path = f"outputs/submissions/submission_openai_{out_suffix}.csv"
    save_submission(test, pred, path)
    print(f"  {name}: CV AUC = {mean_auc:.4f} ± {std_auc:.4f} → {path}")
    return mean_auc


def main():
    print("OpenAI embedding で 3 パターンの提出ファイルを作成します。", flush=True)
    print("データ読み込み（1回）...", flush=True)
    train, test = get_baseline_data()
    y = train["target"].values
    print(f"  train={len(train)}, test={len(test)}", flush=True)

    # --- パターン1: ベースライン38 + OpenAI 16次元 ---
    print("\n[1/3] ベースライン38 + OpenAI 16次元", flush=True)
    train1 = train.copy()
    test1 = test.copy()
    openai_names_16 = add_openai_embedding_features(train1, test1, n_components=16)
    if not openai_names_16:
        raise SystemExit("OpenAI embedding の読み込みに失敗しました。")
    feats1 = BASELINE_FEATURES + openai_names_16
    run_one(train1, test1, y, feats1, "baseline38 + openai_16", "pattern1_baseline38_openai16")

    # --- パターン2: ベースライン38 + OpenAI 8次元 ---
    print("\n[2/3] ベースライン38 + OpenAI 8次元", flush=True)
    train2 = train.copy()
    test2 = test.copy()
    openai_names_8 = add_openai_embedding_features(train2, test2, n_components=8)
    feats2 = BASELINE_FEATURES + openai_names_8
    run_one(train2, test2, y, feats2, "baseline38 + openai_8", "pattern2_baseline38_openai8")

    # --- パターン3: v2（OOF TE+カウント）+ OpenAI 16次元 ---
    print("\n[3/3] v2（OOF TE+カウント）+ OpenAI 16次元", flush=True)
    train3 = train.copy()
    test3 = test.copy()
    add_v2_features(train3, test3, y)
    openai_names_16_v2 = add_openai_embedding_features(train3, test3, n_components=16)
    feats3 = BASELINE_FEATURES + V2_EXTRA_NAMES + openai_names_16_v2
    run_one(train3, test3, y, feats3, "v2 + openai_16", "pattern3_v2_openai16")

    print("\n3本の提出ファイルを作成しました。", flush=True)


if __name__ == "__main__":
    main()
