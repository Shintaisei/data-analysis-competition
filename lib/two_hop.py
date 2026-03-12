"""
2-hop 集約を BPR ベースに追加して実験するためのモジュール。

ベース: submission_atmacup_implicit_bpr16.csv（0.76101）と同じ BPR 16 埋め込み。
ここに「その映画をレビューした批評家たち」の集約特徴（2-hop）を 1 列ずつ追加し、
スコアの増減を見ながら実験できるようにする。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import lightgbm as lgb

from .improvement_candidates import ImprovementContext, get_bpr_base
from .submission import save_submission, verify_submission

# 2-hop で追加する列の名前（add_2hop_features が生成する列と一致させる）
TWO_HOP_FRESH_MEAN = "movie_fresh_rate_mean"
TWO_HOP_CRITIC_TE_MEAN = "movie_critic_te_mean"
TWO_HOP_REVIEW_COUNT = "movie_review_count"
TWO_HOP_COLUMNS = [TWO_HOP_FRESH_MEAN, TWO_HOP_CRITIC_TE_MEAN, TWO_HOP_REVIEW_COUNT]


def add_2hop_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    columns: list[str] | None = None,
) -> list[str]:
    """映画ごとに train で集約した 2-hop 特徴を train_df / test_df に追加する（in place）。
    追加する列: movie_fresh_rate_mean, movie_critic_te_mean, movie_review_count。
    columns で指定した列だけ追加する。None のときは TWO_HOP_COLUMNS すべて。
    Returns:
        追加した列名のリスト。
    """
    if columns is None:
        columns = list(TWO_HOP_COLUMNS)
    else:
        for c in columns:
            if c not in TWO_HOP_COLUMNS:
                raise ValueError(f"不明な 2-hop 列: {c}. 利用可能: {TWO_HOP_COLUMNS}")

    link_col = "rotten_tomatoes_link"
    if "target" not in train_df.columns or link_col not in train_df.columns:
        raise ValueError("train_df に target と rotten_tomatoes_link が必要です")
    if "critic_name_te_ts" not in train_df.columns and TWO_HOP_CRITIC_TE_MEAN in columns:
        raise ValueError("movie_critic_te_mean には train_df に critic_name_te_ts が必要です")

    # 映画ごとに train で集約（その映画をレビューした行の統計）
    by_movie = train_df.groupby(link_col).agg(
        movie_fresh_rate_mean=("target", "mean"),
        movie_critic_te_mean=("critic_name_te_ts", "mean"),
        movie_review_count=("target", "count"),
    ).reset_index()

    # 指定した列だけ追加。欠損は train に登場しなかった映画 → global で埋める
    for c in columns:
        if c not in by_movie.columns:
            continue
        global_val = float(by_movie[c].mean())
        train_df[c] = train_df[link_col].map(by_movie.set_index(link_col)[c]).fillna(global_val).values
        test_df[c] = test_df[link_col].map(by_movie.set_index(link_col)[c]).fillna(global_val).values

    return columns


def run_experiment(
    ctx: ImprovementContext,
    experiment_name: str,
    use_2hop_cols: list[str] | None = None,
    bpr_factors: int = 16,
) -> dict[str, Any]:
    """BPR ベースで学習し、指定した 2-hop 列を追加して提出 CSV を保存する。
    use_2hop_cols が None または空なら BPR のみ（ベースライン再現）。
    use_2hop_cols に TWO_HOP_COLUMNS の名前を並べると、その列だけ追加する。
    Returns:
        verify_submission の返り値（path, ok, message など）。
    """
    train_df, test_df, feats = get_bpr_base(ctx, factors=bpr_factors)
    if use_2hop_cols:
        added = add_2hop_features(train_df, test_df, columns=use_2hop_cols)
        feats = feats + added

    X_tr = train_df[feats]
    X_te = test_df[feats]
    for col in X_tr.columns:
        if not pd.api.types.is_numeric_dtype(X_tr[col]) and not pd.api.types.is_categorical_dtype(X_tr[col]):
            X_tr[col] = X_tr[col].astype("category")
            X_te[col] = X_te[col].astype("category")
    model = lgb.LGBMClassifier(**ctx.lgb_params)
    model.fit(X_tr, ctx.y)
    pred = model.predict_proba(X_te)[:, 1]
    path = ctx.submissions_dir / f"submission_2hop_{experiment_name}.csv"
    save_submission(ctx.test, pred, path, sanitize=True)
    return verify_submission(path, ctx.test)