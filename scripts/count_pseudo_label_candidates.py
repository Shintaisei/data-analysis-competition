#!/usr/bin/env python3
"""
ベースラインで test を予測し、「確信度高め」の疑似ラベル候補が何件あるか集計する。
実行: プロジェクトルートで .venv/bin/python scripts/count_pseudo_label_candidates.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import lightgbm as lgb

from lib import get_baseline_data, BASELINE_FEATURES, BASELINE_LGB_PARAMS


def main():
    print("データ読み込み...", flush=True)
    train, test = get_baseline_data()
    y = train["target"].values
    X = train[BASELINE_FEATURES]
    X_test = test[BASELINE_FEATURES]
    n_train = len(train)
    n_test = len(test)

    print("ベースライン 1 本で全学習 → test 予測...", flush=True)
    model = lgb.LGBMClassifier(**BASELINE_LGB_PARAMS)
    model.fit(X, y)
    pred = model.predict_proba(X_test)[:, 1]

    # 閾値別に件数集計
    thresholds = [
        (0.95, 0.05),
        (0.90, 0.10),
        (0.85, 0.15),
        (0.80, 0.20),
    ]
    print(f"\nテスト件数: {n_test:,}")
    print("閾値（高確信度のみを疑似ラベルとする場合の追加件数）:\n")
    for th_high, th_low in thresholds:
        n_high = (pred >= th_high).sum()
        n_low = (pred <= th_low).sum()
        n_total = n_high + n_low
        pct = 100.0 * n_total / n_test
        print(f"  pred >= {th_high} または <= {th_low}: {n_total:,} 件 ({pct:.1f}%)  [高: {n_high:,}, 低: {n_low:,}]")

    # 学習データに足した場合の増加率
    print(f"\n現在の学習データ: {n_train:,} 件")
    for th_high, th_low in thresholds:
        n_total = (pred >= th_high).sum() + (pred <= th_low).sum()
        new_total = n_train + n_total
        print(f"  閾値 {th_high}/{th_low} で追加すると: {n_train:,} + {n_total:,} = {new_total:,} 件 ({100.0 * n_total / n_train:.1f}% 増)")


if __name__ == "__main__":
    main()
