"""
2-hop 集約を BPR ベースに追加して実験するためのモジュール。

ベース: submission_atmacup_implicit_bpr16.csv（0.76101）と同じ BPR 16 埋め込み。
ここに「その映画をレビューした批評家たち」の集約特徴（2-hop）を 1 列ずつ追加し、
スコアの増減を見ながら実験できるようにする。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import lightgbm as lgb

from .improvement_candidates import ImprovementContext, get_bpr_base
from .submission import save_submission, verify_submission

# 2-hop で追加する列の名前（add_2hop_features が生成する列と一致させる）
TWO_HOP_FRESH_MEAN = "movie_fresh_rate_mean"
TWO_HOP_CRITIC_TE_MEAN = "movie_critic_te_mean"
TWO_HOP_REVIEW_COUNT = "movie_review_count"
TWO_HOP_COLUMNS = [TWO_HOP_FRESH_MEAN, TWO_HOP_CRITIC_TE_MEAN, TWO_HOP_REVIEW_COUNT]
# 2-hop の割り算: 批評家の TE / その映画をレビューした批評家の TE 平均（1位で効いた「値/ユーザー平均」の映画版）
TWO_HOP_CRITIC_TE_DIV_MOVIE_TE = "critic_te_div_movie_te"
BPR_DOT_COL = "bpr_dot"


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
    # link_col が Categorical だと map 結果も Categorical になり fillna(float) で TypeError になるため、
    # キーを str に揃えて map し、結果を ndarray float で欠損補完してから代入する
    by_movie_idx = by_movie.copy()
    by_movie_idx[link_col] = by_movie_idx[link_col].astype(str)
    by_movie_idx = by_movie_idx.set_index(link_col)
    for c in columns:
        if c not in by_movie.columns:
            continue
        global_val = float(by_movie[c].mean())
        ser = by_movie_idx[c].astype(np.float64)
        train_keys = train_df[link_col].astype(str)
        test_keys = test_df[link_col].astype(str)
        train_arr = train_keys.map(ser).to_numpy(dtype=np.float64, na_value=np.nan)
        test_arr = test_keys.map(ser).to_numpy(dtype=np.float64, na_value=np.nan)
        np.nan_to_num(train_arr, nan=global_val, copy=False)
        np.nan_to_num(test_arr, nan=global_val, copy=False)
        train_df[c] = train_arr
        test_df[c] = test_arr

    return columns


def add_bpr_dot_column(train_df: pd.DataFrame, test_df: pd.DataFrame, factors: int = 16) -> None:
    """BPR 埋め込みの内積（user_vec・item_vec）を 1 列 bpr_dot として追加する（in place）。
    get_bpr_base で得た DataFrame に implicit_bpr_{factors}_u_* と implicit_bpr_{factors}_i_* が存在すること。
    """
    u_cols = [f"implicit_bpr_{factors}_u_{k}" for k in range(factors)]
    i_cols = [f"implicit_bpr_{factors}_i_{k}" for k in range(factors)]
    for c in u_cols + i_cols:
        if c not in train_df.columns:
            raise ValueError(f"BPR 列がありません: {c}. get_bpr_base(ctx, factors={factors}) で取得してください。")
    train_df[BPR_DOT_COL] = (train_df[u_cols].values * train_df[i_cols].values).sum(axis=1).astype("float32")
    test_df[BPR_DOT_COL] = (test_df[u_cols].values * test_df[i_cols].values).sum(axis=1).astype("float32")


def add_2hop_ratio_feature(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """2-hop の割り算 1 列を追加: critic_name_te_ts / movie_critic_te_mean。
    add_2hop_features で movie_critic_te_mean を追加したあとで呼ぶこと。0 除算は 1.0 で埋める。
    """
    if TWO_HOP_CRITIC_TE_MEAN not in train_df.columns:
        raise ValueError("先に add_2hop_features で movie_critic_te_mean を追加してください")
    den = train_df[TWO_HOP_CRITIC_TE_MEAN].replace(0, float("nan")).fillna(1.0)
    train_df[TWO_HOP_CRITIC_TE_DIV_MOVIE_TE] = (train_df["critic_name_te_ts"].astype(float) / den).fillna(1.0).values
    den_te = test_df[TWO_HOP_CRITIC_TE_MEAN].replace(0, float("nan")).fillna(1.0)
    test_df[TWO_HOP_CRITIC_TE_DIV_MOVIE_TE] = (test_df["critic_name_te_ts"].astype(float) / den_te).fillna(1.0).values


def run_experiment(
    ctx: ImprovementContext,
    experiment_name: str,
    use_2hop_cols: list[str] | None = None,
    bpr_factors: int = 16,
    use_bpr_dot: bool = False,
    use_2hop_ratio: bool = False,
    bpr_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """BPR ベースで学習し、指定した 2-hop 列・内積 1 列・2-hop 割り算を追加して提出 CSV を保存する。
    use_2hop_cols が None または空なら 2-hop は追加しない。
    use_bpr_dot=True で BPR の内積 1 列を追加。use_2hop_ratio=True で critic_te / movie_critic_te_mean を追加（movie_critic_te_mean が既にあること）。
    bpr_kwargs で BPR の regularization, iterations 等を渡せる（軸1実験用）。
    Returns:
        verify_submission の返り値（path, ok, message など）。
    """
    train_df, test_df, feats = get_bpr_base(ctx, factors=bpr_factors, bpr_kwargs=bpr_kwargs)
    if use_2hop_cols:
        added = add_2hop_features(train_df, test_df, columns=use_2hop_cols)
        feats = feats + added
    if use_2hop_ratio:
        if TWO_HOP_CRITIC_TE_MEAN not in train_df.columns:
            add_2hop_features(train_df, test_df, columns=[TWO_HOP_CRITIC_TE_MEAN])
            feats = feats + [TWO_HOP_CRITIC_TE_MEAN]
        add_2hop_ratio_feature(train_df, test_df)
        feats = feats + [TWO_HOP_CRITIC_TE_DIV_MOVIE_TE]
    if use_bpr_dot:
        add_bpr_dot_column(train_df, test_df, factors=bpr_factors)
        feats = feats + [BPR_DOT_COL]

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


def run_experiment_dot_only(
    ctx: ImprovementContext,
    experiment_name: str,
    bpr_factors: int = 16,
) -> dict[str, Any]:
    """BPR の内積 1 列だけを 55 特徴に足して学習（埋め込み 32 列は使わない）。CF の「この組み合わせのスコア」を直接 LGB に渡す実験。"""
    train_df, test_df, _ = get_bpr_base(ctx, factors=bpr_factors)
    add_bpr_dot_column(train_df, test_df, factors=bpr_factors)
    feats = ctx.features + [BPR_DOT_COL]
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