"""
過去コンペ（atmaCup #15 等）の上位解法をそのまま再現し、
複数パターンの提出ファイルを生成するモジュール。

- OOF TE + カウント（5位: Target Encoding）
- 比率特徴量（1位・57位: 集約値で割る）
- NMF（3位: 批評家×映画の行列分解）
- CountVectorizer + SVD（2位: id2vec 風）
- 既知/未知で二モデル（1位・57位: seen/unseen 別モデル、GroupKFold）
- シードアベレージング（3位・57位: 複数 seed の予測平均）
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, GroupKFold
import lightgbm as lgb


# ---------------------------------------------------------------------------
# 定数（参照: docs/PROJECT_MIND.md）
# ---------------------------------------------------------------------------

VAL_YEARS = [2013, 2014, 2015, 2016]
N_FOLDS_TE = 5
NMF_N_COMP = 24
SVD_N_COMP = 32
SEED_AVERAGING_SEEDS = [42, 43, 44]  # 57位・3位: シードアベレージング

# train_baseline.ipynb と同じパラメータを lib から参照（チューニング時は lib.BASELINE_LGB_PARAMS を更新）
try:
    from lib import BASELINE_LGB_PARAMS as DEFAULT_LGB_PARAMS
except ImportError:
    DEFAULT_LGB_PARAMS = {
        "objective": "binary", "metric": "auc", "boosting_type": "gbdt",
        "random_state": 42, "verbosity": -1,
        "n_estimators": 1011, "learning_rate": 0.04392, "num_leaves": 74,
        "min_data_in_leaf": 46, "feature_fraction": 0.7514,
        "reg_alpha": 0.9285, "reg_lambda": 4.2552, "max_depth": 10,
    }

TE_CONFIG = [
    ("critic_name", 10),
    ("rotten_tomatoes_link", 10),
    ("publisher_name", 10),
    ("directors", 15),
    ("authors", 15),
    ("production_company", 10),
]

V2_EXTRA_NAMES = [
    "critic_name_te",
    "rotten_tomatoes_link_te",
    "publisher_name_te",
    "directors_te",
    "authors_te",
    "production_company_te",
    "critic_review_count",
    "movie_review_count",
]

RATIO_FEATURE_NAMES = [
    "critic_avg_fresh",
    "ratio_movie_te_to_critic",
    "ratio_movie_te_to_global",
    "ratio_critic_te_to_global",
]


# ---------------------------------------------------------------------------
# OOF Target Encoding + カウント（5位: TE が効いていた）
# ---------------------------------------------------------------------------

def _target_encode_oof(
    train_col: pd.Series,
    y: np.ndarray,
    fold_indices: list[tuple[np.ndarray, np.ndarray]],
    smoothing: float = 10,
) -> pd.Series:
    oof_te = pd.Series(np.nan, index=train_col.index)
    gm = float(np.mean(y))
    for train_idx, val_idx in fold_indices:
        fd = pd.DataFrame({"col": train_col.iloc[train_idx], "target": y[train_idx]})
        st = fd.groupby("col", observed=True)["target"].agg(["mean", "count"])
        st["te"] = (st["count"] * st["mean"] + smoothing * gm) / (st["count"] + smoothing)
        oof_te.iloc[val_idx] = train_col.iloc[val_idx].map(st["te"].to_dict()).fillna(gm)
    return oof_te


def _target_encode_test(
    train_col: pd.Series,
    y: np.ndarray,
    test_col: pd.Series,
    smoothing: float = 10,
) -> pd.Series:
    gm = float(np.mean(y))
    st = pd.DataFrame({"col": train_col, "target": y}).groupby("col", observed=True)["target"].agg(["mean", "count"])
    st["te"] = (st["count"] * st["mean"] + smoothing * gm) / (st["count"] + smoothing)
    return test_col.map(st["te"].to_dict()).fillna(gm)


def add_v2_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: np.ndarray,
    n_splits: int = N_FOLDS_TE,
    te_config: list[tuple[str, float]] | None = None,
) -> list[str]:
    """OOF TE 6本 + カウント2本を追加。atmaCup 5位の Target Encoding を再現。"""
    te_config = te_config or TE_CONFIG
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_te = list(kf.split(train))
    for col, sm in te_config:
        tr_s = train[col].astype(str).fillna("missing")
        te_s = test[col].astype(str).fillna("missing")
        train[f"{col}_te"] = _target_encode_oof(tr_s, y, fold_te, sm)
        test[f"{col}_te"] = _target_encode_test(tr_s, y, te_s, sm)
    critic_counts = train.groupby("critic_name", observed=True).size()
    train["critic_review_count"] = train["critic_name"].map(critic_counts).astype(float)
    test["critic_review_count"] = test["critic_name"].map(critic_counts).astype(float).fillna(0)
    movie_counts = train.groupby("rotten_tomatoes_link", observed=True).size()
    train["movie_review_count"] = train["rotten_tomatoes_link"].map(movie_counts).astype(float)
    test["movie_review_count"] = test["rotten_tomatoes_link"].map(movie_counts).astype(float).fillna(0)
    return list(V2_EXTRA_NAMES)


# ---------------------------------------------------------------------------
# 比率特徴量（1位・57位: 集約前の値 ÷ 集約結果）
# ---------------------------------------------------------------------------

def add_ratio_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    global_mean: float,
) -> list[str]:
    """「値 ÷ 集約結果」の比率特徴量。1位・57位で効いていた差より割り算。"""
    critic_avg = train.groupby("critic_name", observed=True)["target"].mean().to_dict()
    train["critic_avg_fresh"] = train["critic_name"].map(critic_avg).fillna(global_mean)
    test["critic_avg_fresh"] = test["critic_name"].map(critic_avg).fillna(global_mean)
    train["ratio_movie_te_to_critic"] = (
        (train["rotten_tomatoes_link_te"] / train["critic_avg_fresh"])
        .replace([np.inf, -np.inf], np.nan)
        .fillna(1.0)
    )
    test["ratio_movie_te_to_critic"] = (
        (test["rotten_tomatoes_link_te"] / test["critic_avg_fresh"])
        .replace([np.inf, -np.inf], np.nan)
        .fillna(1.0)
    )
    train["ratio_movie_te_to_global"] = train["rotten_tomatoes_link_te"] / global_mean
    test["ratio_movie_te_to_global"] = test["rotten_tomatoes_link_te"] / global_mean
    train["ratio_critic_te_to_global"] = train["critic_name_te"] / global_mean
    test["ratio_critic_te_to_global"] = test["critic_name_te"] / global_mean
    return list(RATIO_FEATURE_NAMES)


# ---------------------------------------------------------------------------
# NMF（3位: 批評家×映画の行列分解）
# ---------------------------------------------------------------------------

def add_nmf_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    n_components: int = NMF_N_COMP,
    random_state: int = 42,
) -> list[str]:
    """行=批評家・列=映画の 0/1 行列を NMF 分解し、埋め込みを特徴量に。3位解法の再現。"""
    train_uniq = train.drop_duplicates(subset=["critic_name", "rotten_tomatoes_link"])
    critics = train_uniq["critic_name"].astype(str).unique()
    movies = train_uniq["rotten_tomatoes_link"].astype(str).unique()
    c2i = {str(c): i for i, c in enumerate(critics)}
    m2j = {str(m): j for j, m in enumerate(movies)}
    rows, cols, data = [], [], []
    for _, r in train_uniq.iterrows():
        ci = c2i.get(str(r["critic_name"]))
        mj = m2j.get(str(r["rotten_tomatoes_link"]))
        if ci is not None and mj is not None:
            rows.append(ci)
            cols.append(mj)
            data.append(1.0)
    M = csr_matrix((data, (rows, cols)), shape=(len(critics), len(movies)))
    nmf = NMF(n_components=n_components, random_state=random_state, max_iter=200)
    W = nmf.fit_transform(M)
    H = nmf.components_
    mean_H = float(np.mean(H))
    names = []
    for k in range(n_components):
        train[f"nmf_c_{k}"] = train["critic_name"].astype(str).map(
            lambda c, kk=k: W[c2i[c], kk] if c in c2i else 0.0
        )
        test[f"nmf_c_{k}"] = test["critic_name"].astype(str).map(
            lambda c, kk=k: W[c2i[c], kk] if c in c2i else 0.0
        )
        names.append(f"nmf_c_{k}")
    for k in range(n_components):
        train[f"nmf_m_{k}"] = train["rotten_tomatoes_link"].astype(str).map(
            lambda m, kk=k: H[kk, m2j[m]] if m in m2j else mean_H
        )
        test[f"nmf_m_{k}"] = test["rotten_tomatoes_link"].astype(str).map(
            lambda m, kk=k: H[kk, m2j[m]] if m in m2j else mean_H
        )
        names.append(f"nmf_m_{k}")
    return names


# ---------------------------------------------------------------------------
# CountVectorizer + TruncatedSVD（2位: id2vec 風）
# ---------------------------------------------------------------------------

def add_openai_embedding_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    path: str | None = None,
    n_components: int = 16,
    random_state: int = 42,
) -> list[str]:
    """
    保存済みの OpenAI movie_info embedding を読み込み、PCA で圧縮して特徴量として追加する。
    先に run_openai_embeddings_once.py を1回実行して outputs/embeddings/movie_info_embeddings.parquet を作成すること。
    ファイルが無い場合は何も追加せず [] を返す。
    """
    try:
        from lib.openai_embeddings import load_movie_info_embeddings
    except ImportError:
        return []
    try:
        emb_df = load_movie_info_embeddings(path=path)
    except FileNotFoundError:
        return []
    emb_cols = [c for c in emb_df.columns if c != "rotten_tomatoes_link"]
    # 左結合で embedding を付与（行順は train/test のまま）
    train_emb = train[["rotten_tomatoes_link"]].merge(
        emb_df[["rotten_tomatoes_link"] + emb_cols],
        on="rotten_tomatoes_link",
        how="left",
    )
    test_emb = test[["rotten_tomatoes_link"]].merge(
        emb_df[["rotten_tomatoes_link"] + emb_cols],
        on="rotten_tomatoes_link",
        how="left",
    )
    X_tr = train_emb[emb_cols].fillna(0).astype(np.float32).values
    X_te = test_emb[emb_cols].fillna(0).astype(np.float32).values
    from sklearn.decomposition import PCA
    n_comp = min(n_components, len(emb_cols), X_tr.shape[1])
    pca = PCA(n_components=n_comp, random_state=random_state)
    tr_pca = pca.fit_transform(X_tr)
    te_pca = pca.transform(X_te)
    names = [f"openai_{i}" for i in range(n_comp)]
    for i, name in enumerate(names):
        train[name] = tr_pca[:, i]
        test[name] = te_pca[:, i]
    return names


def add_svd_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    n_components: int = SVD_N_COMP,
    random_state: int = 42,
) -> list[str]:
    """批評家ごとの映画ID列・映画ごとの批評家ID列を document に CountVectorizer → SVD。2位解法の再現。"""
    critic_docs = train.groupby("critic_name", observed=True)["rotten_tomatoes_link"].apply(
        lambda x: " ".join(x.astype(str))
    ).to_dict()
    all_c = list(critic_docs.keys())
    doc_list = [critic_docs.get(c, "") for c in all_c]
    cv = CountVectorizer(max_features=2000, token_pattern=r"\S+")
    X_cv = cv.fit_transform(doc_list)
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    C_emb = svd.fit_transform(X_cv)
    c2idx = {str(c): i for i, c in enumerate(all_c)}

    movie_docs = train.groupby("rotten_tomatoes_link", observed=True)["critic_name"].apply(
        lambda x: " ".join(x.astype(str))
    ).to_dict()
    all_m = list(movie_docs.keys())
    doc_list_m = [movie_docs.get(m, "") for m in all_m]
    cv_m = CountVectorizer(max_features=2000, token_pattern=r"\S+")
    X_cv_m = cv_m.fit_transform(doc_list_m)
    svd_m = TruncatedSVD(n_components=n_components, random_state=random_state)
    M_emb = svd_m.fit_transform(X_cv_m)
    m2idx = {str(m): i for i, m in enumerate(all_m)}

    names = []
    for k in range(n_components):
        train[f"svd_c_{k}"] = train["critic_name"].astype(str).map(
            lambda c, kk=k: C_emb[c2idx[c], kk] if c in c2idx else 0.0
        )
        test[f"svd_c_{k}"] = test["critic_name"].astype(str).map(
            lambda c, kk=k: C_emb[c2idx[c], kk] if c in c2idx else 0.0
        )
        names.append(f"svd_c_{k}")
    for k in range(n_components):
        mean_mk = float(np.mean(M_emb[:, k]))
        train[f"svd_m_{k}"] = train["rotten_tomatoes_link"].astype(str).map(
            lambda m, kk=k: M_emb[m2idx[m], kk] if m in m2idx else mean_mk
        )
        test[f"svd_m_{k}"] = test["rotten_tomatoes_link"].astype(str).map(
            lambda m, kk=k: M_emb[m2idx[m], kk] if m in m2idx else mean_mk
        )
        names.append(f"svd_m_{k}")
    return names


# ---------------------------------------------------------------------------
# 時間整合（リークなし）: tr_idx より前のデータだけで TE・カウントを計算
# ---------------------------------------------------------------------------

def _build_v2_from_tr(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: np.ndarray,
    tr_idx: np.ndarray,
    val_idx: np.ndarray,
    te_config: list[tuple[str, float]] | None = None,
    smoothing: float = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """tr_idx のデータだけで TE とカウントを計算し、tr/val/test 用の DataFrame（8列）を返す。"""
    te_config = te_config or TE_CONFIG
    tr_df = train.iloc[tr_idx]
    y_tr = y[tr_idx]
    gm = float(np.mean(y_tr))
    out_tr = pd.DataFrame(index=tr_idx)
    out_val = pd.DataFrame(index=val_idx)
    out_test = pd.DataFrame(index=test.index)

    for col, sm in te_config:
        st = pd.DataFrame({"col": tr_df[col].astype(str).fillna("missing"), "target": y_tr}).groupby("col", observed=True)["target"].agg(["mean", "count"])
        st["te"] = (st["count"] * st["mean"] + sm * gm) / (st["count"] + sm)
        map_te = st["te"].to_dict()
        out_tr[f"{col}_te"] = tr_df[col].astype(str).fillna("missing").map(map_te).fillna(gm).values
        out_val[f"{col}_te"] = train.iloc[val_idx][col].astype(str).fillna("missing").map(map_te).fillna(gm).values
        out_test[f"{col}_te"] = test[col].astype(str).fillna("missing").map(map_te).fillna(gm).values

    critic_counts = tr_df.groupby("critic_name", observed=True).size()
    movie_counts = tr_df.groupby("rotten_tomatoes_link", observed=True).size()
    out_tr["critic_review_count"] = tr_df["critic_name"].map(critic_counts).astype(float).fillna(0).values
    out_val["critic_review_count"] = train.iloc[val_idx]["critic_name"].map(critic_counts).astype(float).fillna(0).values
    out_test["critic_review_count"] = test["critic_name"].map(critic_counts).astype(float).fillna(0).values
    out_tr["movie_review_count"] = tr_df["rotten_tomatoes_link"].map(movie_counts).astype(float).fillna(0).values
    out_val["movie_review_count"] = train.iloc[val_idx]["rotten_tomatoes_link"].map(movie_counts).astype(float).fillna(0).values
    out_test["movie_review_count"] = test["rotten_tomatoes_link"].map(movie_counts).astype(float).fillna(0).values

    return out_tr, out_val, out_test


def _build_ratio_from_tr(
    train: pd.DataFrame,
    test: pd.DataFrame,
    val_idx: np.ndarray,
    out_tr: pd.DataFrame,
    out_val: pd.DataFrame,
    out_test: pd.DataFrame,
    tr_idx: np.ndarray,
    global_mean: float,
) -> None:
    """out_* に比率特徴量 4 列を追加。tr の統計のみ使用。"""
    tr_df = train.iloc[tr_idx]
    critic_avg = tr_df.groupby("critic_name", observed=True)["target"].mean().to_dict()
    out_tr["critic_avg_fresh"] = train.iloc[tr_idx]["critic_name"].map(critic_avg).fillna(global_mean).values
    out_tr["ratio_movie_te_to_critic"] = (out_tr["rotten_tomatoes_link_te"] / out_tr["critic_avg_fresh"]).replace([np.inf, -np.inf], np.nan).fillna(1.0).values
    out_tr["ratio_movie_te_to_global"] = (out_tr["rotten_tomatoes_link_te"] / global_mean).values
    out_tr["ratio_critic_te_to_global"] = (out_tr["critic_name_te"] / global_mean).values
    out_val["critic_avg_fresh"] = train.iloc[val_idx]["critic_name"].map(critic_avg).fillna(global_mean).values
    out_val["ratio_movie_te_to_critic"] = (out_val["rotten_tomatoes_link_te"] / out_val["critic_avg_fresh"]).replace([np.inf, -np.inf], np.nan).fillna(1.0).values
    out_val["ratio_movie_te_to_global"] = (out_val["rotten_tomatoes_link_te"] / global_mean).values
    out_val["ratio_critic_te_to_global"] = (out_val["critic_name_te"] / global_mean).values
    out_test["critic_avg_fresh"] = test["critic_name"].map(critic_avg).fillna(global_mean).values
    out_test["ratio_movie_te_to_critic"] = (out_test["rotten_tomatoes_link_te"] / out_test["critic_avg_fresh"]).replace([np.inf, -np.inf], np.nan).fillna(1.0).values
    out_test["ratio_movie_te_to_global"] = (out_test["rotten_tomatoes_link_te"] / global_mean).values
    out_test["ratio_critic_te_to_global"] = (out_test["critic_name_te"] / global_mean).values


def _build_nmf_from_tr(
    train: pd.DataFrame,
    test: pd.DataFrame,
    tr_idx: np.ndarray,
    val_idx: np.ndarray,
    n_components: int = NMF_N_COMP,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """tr_idx のデータだけで NMF を fit し、tr/val/test 用の NMF 特徴 DataFrame を返す。"""
    tr_df = train.iloc[tr_idx]
    train_uniq = tr_df.drop_duplicates(subset=["critic_name", "rotten_tomatoes_link"])
    critics = train_uniq["critic_name"].astype(str).unique()
    movies = train_uniq["rotten_tomatoes_link"].astype(str).unique()
    c2i = {str(c): i for i, c in enumerate(critics)}
    m2j = {str(m): j for j, m in enumerate(movies)}
    rows, cols, data = [], [], []
    for _, r in train_uniq.iterrows():
        ci = c2i.get(str(r["critic_name"]))
        mj = m2j.get(str(r["rotten_tomatoes_link"]))
        if ci is not None and mj is not None:
            rows.append(ci)
            cols.append(mj)
            data.append(1.0)
    M = csr_matrix((data, (rows, cols)), shape=(len(critics), len(movies)))
    nmf = NMF(n_components=n_components, random_state=random_state, max_iter=200)
    W = nmf.fit_transform(M)
    H = nmf.components_
    mean_H = float(np.mean(H))
    names_c = [f"nmf_c_{k}" for k in range(n_components)]
    names_m = [f"nmf_m_{k}" for k in range(n_components)]
    out_tr = pd.DataFrame(index=tr_idx)
    out_val = pd.DataFrame(index=val_idx)
    out_test = pd.DataFrame(index=test.index)
    for k in range(n_components):
        out_tr[f"nmf_c_{k}"] = tr_df["critic_name"].astype(str).map(lambda c, kk=k: W[c2i[c], kk] if c in c2i else 0.0).values
        out_val[f"nmf_c_{k}"] = train.iloc[val_idx]["critic_name"].astype(str).map(lambda c, kk=k: W[c2i[c], kk] if c in c2i else 0.0).values
        out_test[f"nmf_c_{k}"] = test["critic_name"].astype(str).map(lambda c, kk=k: W[c2i[c], kk] if c in c2i else 0.0).values
    for k in range(n_components):
        out_tr[f"nmf_m_{k}"] = tr_df["rotten_tomatoes_link"].astype(str).map(lambda m, kk=k: H[kk, m2j[m]] if m in m2j else mean_H).values
        out_val[f"nmf_m_{k}"] = train.iloc[val_idx]["rotten_tomatoes_link"].astype(str).map(lambda m, kk=k: H[kk, m2j[m]] if m in m2j else mean_H).values
        out_test[f"nmf_m_{k}"] = test["rotten_tomatoes_link"].astype(str).map(lambda m, kk=k: H[kk, m2j[m]] if m in m2j else mean_H).values
    return out_tr, out_val, out_test


def _build_svd_from_tr(
    train: pd.DataFrame,
    test: pd.DataFrame,
    tr_idx: np.ndarray,
    val_idx: np.ndarray,
    n_components: int = SVD_N_COMP,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """tr_idx のデータだけで CountVec+SVD を fit し、tr/val/test 用の SVD 特徴 DataFrame を返す。"""
    tr_df = train.iloc[tr_idx]
    critic_docs = tr_df.groupby("critic_name", observed=True)["rotten_tomatoes_link"].apply(lambda x: " ".join(x.astype(str))).to_dict()
    all_c = list(critic_docs.keys())
    cv = CountVectorizer(max_features=2000, token_pattern=r"\S+")
    X_cv = cv.fit_transform([critic_docs.get(c, "") for c in all_c])
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    C_emb = svd.fit_transform(X_cv)
    c2idx = {str(c): i for i, c in enumerate(all_c)}
    movie_docs = tr_df.groupby("rotten_tomatoes_link", observed=True)["critic_name"].apply(lambda x: " ".join(x.astype(str))).to_dict()
    all_m = list(movie_docs.keys())
    cv_m = CountVectorizer(max_features=2000, token_pattern=r"\S+")
    X_cv_m = cv_m.fit_transform([movie_docs.get(m, "") for m in all_m])
    svd_m = TruncatedSVD(n_components=n_components, random_state=random_state)
    M_emb = svd_m.fit_transform(X_cv_m)
    m2idx = {str(m): i for i, m in enumerate(all_m)}
    out_tr = pd.DataFrame(index=tr_idx)
    out_val = pd.DataFrame(index=val_idx)
    out_test = pd.DataFrame(index=test.index)
    for k in range(n_components):
        out_tr[f"svd_c_{k}"] = tr_df["critic_name"].astype(str).map(lambda c, kk=k: C_emb[c2idx[c], kk] if c in c2idx else 0.0).values
        out_val[f"svd_c_{k}"] = train.iloc[val_idx]["critic_name"].astype(str).map(lambda c, kk=k: C_emb[c2idx[c], kk] if c in c2idx else 0.0).values
        out_test[f"svd_c_{k}"] = test["critic_name"].astype(str).map(lambda c, kk=k: C_emb[c2idx[c], kk] if c in c2idx else 0.0).values
    for k in range(n_components):
        mean_mk = float(np.mean(M_emb[:, k]))
        out_tr[f"svd_m_{k}"] = tr_df["rotten_tomatoes_link"].astype(str).map(lambda m, kk=k: M_emb[m2idx[m], kk] if m in m2idx else mean_mk).values
        out_val[f"svd_m_{k}"] = train.iloc[val_idx]["rotten_tomatoes_link"].astype(str).map(lambda m, kk=k: M_emb[m2idx[m], kk] if m in m2idx else mean_mk).values
        out_test[f"svd_m_{k}"] = test["rotten_tomatoes_link"].astype(str).map(lambda m, kk=k: M_emb[m2idx[m], kk] if m in m2idx else mean_mk).values
    return out_tr, out_val, out_test


def build_fold_features_time_aware(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: np.ndarray,
    tr_idx: np.ndarray,
    val_idx: np.ndarray,
    baseline_features: list[str],
    add_ratio: bool = False,
    add_nmf: bool = False,
    add_svd: bool = False,
    global_mean: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """1 fold 分の特徴量を tr のみから作成（時間整合・リークなし）。(X_tr, X_val, X_test, feature_names) を返す。"""
    gm = float(np.mean(y[tr_idx])) if global_mean is None else global_mean
    out_tr_v2, out_val_v2, out_test_v2 = _build_v2_from_tr(train, test, y, tr_idx, val_idx)
    if add_ratio:
        _build_ratio_from_tr(train, test, val_idx, out_tr_v2, out_val_v2, out_test_v2, tr_idx, gm)
    if add_nmf:
        out_tr_n, out_val_n, out_test_n = _build_nmf_from_tr(train, test, tr_idx, val_idx)
        out_tr_v2 = pd.concat([out_tr_v2, out_tr_n], axis=1)
        out_val_v2 = pd.concat([out_val_v2, out_val_n], axis=1)
        out_test_v2 = pd.concat([out_test_v2, out_test_n], axis=1)
    if add_svd:
        out_tr_s, out_val_s, out_test_s = _build_svd_from_tr(train, test, tr_idx, val_idx)
        out_tr_v2 = pd.concat([out_tr_v2, out_tr_s], axis=1)
        out_val_v2 = pd.concat([out_val_v2, out_val_s], axis=1)
        out_test_v2 = pd.concat([out_test_v2, out_test_s], axis=1)
    base_tr = train.iloc[tr_idx][baseline_features].copy()
    base_val = train.iloc[val_idx][baseline_features].copy()
    base_test = test[baseline_features].copy()
    X_tr = pd.concat([base_tr.reset_index(drop=True), out_tr_v2.reset_index(drop=True)], axis=1)
    X_val = pd.concat([base_val.reset_index(drop=True), out_val_v2.reset_index(drop=True)], axis=1)
    X_test = pd.concat([base_test.reset_index(drop=True), out_test_v2.reset_index(drop=True)], axis=1)
    feature_names = list(X_tr.columns)
    return X_tr, X_val, X_test, feature_names


def run_time_series_cv_time_aware(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: np.ndarray,
    time_splits: list[tuple[np.ndarray, np.ndarray]],
    baseline_features: list[str],
    config_key: str,
    lgb_params: dict[str, Any],
    early_stopping_rounds: int = 30,
    verbose: bool = True,
) -> tuple[float, float, list[np.ndarray]]:
    """時間整合で fold ごとに特徴量を構築して時系列CV。config_key: v2, v2_ratio, v2_nmf, v2_svd, v2_all。"""
    add_ratio = config_key in ("v2_ratio", "v2_all")
    add_nmf = config_key in ("v2_nmf", "v2_all")
    add_svd = config_key in ("v2_svd", "v2_all")
    global_mean = float(np.mean(y))
    fold_scores = []
    test_preds = []
    for fold, (tr_idx, val_idx) in enumerate(time_splits):
        X_tr, X_val, X_test, _ = build_fold_features_time_aware(
            train, test, y, tr_idx, val_idx, baseline_features,
            add_ratio=add_ratio, add_nmf=add_nmf, add_svd=add_svd, global_mean=global_mean,
        )
        model = lgb.LGBMClassifier(**lgb_params)
        model.fit(
            X_tr, y[tr_idx],
            eval_set=[(X_val, y[val_idx])],
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
        )
        auc = roc_auc_score(y[val_idx], model.predict_proba(X_val)[:, 1])
        fold_scores.append(auc)
        test_preds.append(model.predict_proba(X_test)[:, 1])
        if verbose:
            print(f"  Fold{fold+1}: AUC={auc:.4f}")
    mean_auc = float(np.mean(fold_scores))
    std_auc = float(np.std(fold_scores))
    return mean_auc, std_auc, test_preds


# ---------------------------------------------------------------------------
# 時系列CV 分割
# ---------------------------------------------------------------------------

def get_time_splits(
    train: pd.DataFrame,
    val_years: list[int] | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """時系列CV: 各 val_year より前で学習・その年で検証。"""
    val_years = val_years or VAL_YEARS
    time_splits = []
    for vy in val_years:
        tr_idx = np.where(train["review_year"] < vy)[0]
        val_idx = np.where(train["review_year"] == vy)[0]
        if len(val_idx) > 0:
            time_splits.append((tr_idx, val_idx))
    return time_splits


# ---------------------------------------------------------------------------
# 時系列CV 実行 + test 予測
# ---------------------------------------------------------------------------

def run_time_series_cv(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    y: np.ndarray,
    time_splits: list[tuple[np.ndarray, np.ndarray]],
    lgb_params: dict[str, Any] | None = None,
    early_stopping_rounds: int = 30,
    verbose: bool = True,
) -> tuple[float, float, list[np.ndarray]]:
    """時系列CV を回し、(mean_auc, std_auc, test_preds_per_fold) を返す。"""
    params = {**DEFAULT_LGB_PARAMS, **(lgb_params or {})}
    X = train[features]
    X_test = test[features]
    fold_scores = []
    test_preds = []
    for fold, (tr_idx, val_idx) in enumerate(time_splits):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
        )
        auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
        fold_scores.append(auc)
        test_preds.append(model.predict_proba(X_test)[:, 1])
        if verbose:
            print(f"  Fold{fold+1}: AUC={auc:.4f}")
    mean_auc = float(np.mean(fold_scores))
    std_auc = float(np.std(fold_scores))
    return mean_auc, std_auc, test_preds


# ---------------------------------------------------------------------------
# 既知/未知で二モデル（1位・57位: seen/unseen 別モデル）
# ---------------------------------------------------------------------------

def run_known_unknown_two_models(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: np.ndarray,
    features_known: list[str],
    features_unknown: list[str],
    time_splits: list[tuple[np.ndarray, np.ndarray]],
    lgb_params: dict[str, Any] | None = None,
    n_folds_unknown: int = 5,
    verbose: bool = True,
) -> tuple[dict[str, float], np.ndarray]:
    """
    既知映画用・未知映画用で別モデルを学習し、test で映画の有無に応じて振り分け予測。
    atmaCup 1位・57位の「seen/unseen で専用モデル」を再現。
    戻り値: (summary_dict, test_pred)
    """
    params = {**DEFAULT_LGB_PARAMS, **(lgb_params or {})}
    train_movies = set(train["rotten_tomatoes_link"])
    test["_is_known_movie"] = test["rotten_tomatoes_link"].isin(train_movies)

    # 未知用: GroupKFold（group=映画）で「検証映画を学習に含めない」
    gkf = GroupKFold(n_splits=n_folds_unknown)
    groups_movie = train["rotten_tomatoes_link"].values
    unknown_splits = list(gkf.split(train, y, groups=groups_movie))

    X_unk = train[features_unknown]
    X_unk_test = test[features_unknown]
    oof_unk = np.zeros(len(train))
    preds_unk = []
    scores_unk = []
    for fold, (tr_idx, val_idx) in enumerate(unknown_splits):
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_unk.iloc[tr_idx], y[tr_idx],
            eval_set=[(X_unk.iloc[val_idx], y[val_idx])],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        oof_unk[val_idx] = model.predict_proba(X_unk.iloc[val_idx])[:, 1]
        auc = roc_auc_score(y[val_idx], oof_unk[val_idx])
        scores_unk.append(auc)
        preds_unk.append(model.predict_proba(X_unk_test)[:, 1])
        if verbose:
            print(f"  未知用 Fold{fold+1}: AUC={auc:.4f}")
    mean_unk = float(np.mean(scores_unk))
    pred_unknown_avg = np.mean(preds_unk, axis=0)

    # 既知用: 時系列CV
    X_known = train[features_known]
    X_known_test = test[features_known]
    _, _, preds_known = run_time_series_cv(
        train, test, features_known, y, time_splits, params,
        verbose=verbose,
    )
    pred_known_avg = np.mean(preds_known, axis=0)

    is_known_test = test["_is_known_movie"].values
    final_pred = np.where(is_known_test, pred_known_avg, pred_unknown_avg)
    test.drop(columns=["_is_known_movie"], inplace=True)

    summary = {"unknown_cv_auc": mean_unk}
    return summary, final_pred


# ---------------------------------------------------------------------------
# シードアベレージング（3位・57位）
# ---------------------------------------------------------------------------

def run_seed_averaging(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    y: np.ndarray,
    time_splits: list[tuple[np.ndarray, np.ndarray]],
    seeds: list[int] | None = None,
    lgb_params: dict[str, Any] | None = None,
    verbose: bool = True,
) -> tuple[float, float, np.ndarray]:
    """複数 seed で学習し、test 予測の平均を返す。3位・57位のシードアベレージング再現。"""
    seeds = seeds or SEED_AVERAGING_SEEDS
    all_preds = []
    all_means = []
    for seed in seeds:
        params = {**(lgb_params or DEFAULT_LGB_PARAMS), "random_state": seed}
        mean_auc, std_auc, test_preds = run_time_series_cv(
            train, test, features, y, time_splits, params, verbose=False
        )
        all_means.append(mean_auc)
        all_preds.append(np.mean(test_preds, axis=0))
    pred_avg = np.mean(all_preds, axis=0)
    mean_auc = float(np.mean(all_means))
    std_auc = float(np.std(all_means))
    if verbose:
        print(f"  Seeds {seeds} → CV AUC mean={mean_auc:.4f} ± {std_auc:.4f}")
    return mean_auc, std_auc, pred_avg


# ---------------------------------------------------------------------------
# 未知用特徴量リスト（映画ID・映画TE・movie_review_count 等を除外）
# ---------------------------------------------------------------------------

def get_unknown_movie_features(
    features_all: list[str],
    n_nmf: int = NMF_N_COMP,
    n_svd: int = SVD_N_COMP,
) -> list[str]:
    """既知用全特徴量から、未知映画で使えないものを除いたリスト。"""
    exclude = {
        "rotten_tomatoes_link",
        "rotten_tomatoes_link_te",
        "movie_review_count",
        "ratio_movie_te_to_critic",
        "ratio_movie_te_to_global",
    }
    feats = [f for f in features_all if f not in exclude]
    feats = [f for f in feats if not f.startswith("nmf_m_") and not f.startswith("svd_m_")]
    feats = feats + [f"nmf_m_{k}" for k in range(n_nmf)] + [f"svd_m_{k}" for k in range(n_svd)]
    return feats


def get_unknown_features_simple(features: list[str]) -> list[str]:
    """
    既知用特徴量リストから、未知映画で使えない列だけ除いたリスト（baseline / v2 など v2_all 以外の段階用）。
    v2_all の未知用は get_unknown_movie_features を使う。
    """
    exclude = {
        "rotten_tomatoes_link",
        "rotten_tomatoes_link_te",
        "movie_review_count",
        "ratio_movie_te_to_critic",
        "ratio_movie_te_to_global",
    }
    return [f for f in features if f not in exclude and not f.startswith("nmf_m_") and not f.startswith("svd_m_")]


# ---------------------------------------------------------------------------
# 全特徴量追加 + 複数提出ファイル生成
# ---------------------------------------------------------------------------

def add_all_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: np.ndarray,
    baseline_features: list[str],
) -> tuple[float, list[tuple[np.ndarray, np.ndarray]], dict[str, list[str]]]:
    """
    v2 / 比率 / NMF / SVD を順に追加する。
    戻り値: (global_mean, time_splits, feature_sets)
    feature_sets は "v2", "ratio", "nmf", "svd", "all", "all_for_unknown" などのキーを持つ。
    """
    global_mean = float(y.mean())
    add_v2_features(train, test, y)
    add_ratio_features(train, test, global_mean)
    nmf_names = add_nmf_features(train, test)
    svd_names = add_svd_features(train, test)

    v2_list = baseline_features + V2_EXTRA_NAMES
    ratio_list = v2_list + RATIO_FEATURE_NAMES
    nmf_list = v2_list + nmf_names
    svd_list = v2_list + svd_names
    all_list = v2_list + RATIO_FEATURE_NAMES + nmf_names + svd_names
    unknown_list = get_unknown_movie_features(all_list, n_nmf=NMF_N_COMP, n_svd=SVD_N_COMP)

    time_splits = get_time_splits(train)
    feature_sets = {
        "baseline": baseline_features,
        "v2": v2_list,
        "v2_ratio": ratio_list,
        "v2_nmf": nmf_list,
        "v2_svd": svd_list,
        "v2_all": all_list,
        "known": all_list,
        "unknown": unknown_list,
    }
    return global_mean, time_splits, feature_sets


def save_submission(
    test: pd.DataFrame,
    pred: np.ndarray,
    path: str,
    id_col: str = "ID",
) -> None:
    """提出用 CSV を保存。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame({id_col: test[id_col], "target": pred}).to_csv(path, index=False)


def verify_submissions(
    test: pd.DataFrame,
    output_dir: str,
    expected_files: list[str] | None = None,
    id_col: str = "ID",
) -> pd.DataFrame:
    """
    提出ファイルが正しく作成されているか検証する。
    戻り値: 各ファイルの path, rows, cols_ok, target_ok, id_ok の DataFrame。
    """
    if expected_files is None:
        expected_files = [
            "submission_baseline38.csv",
            "submission_v2.csv",
            "submission_v2_ratio.csv",
            "submission_v2_nmf.csv",
            "submission_v2_svd.csv",
            "submission_v2_all.csv",
            "submission_known_unknown.csv",
            "submission_v2_all_seed3.csv",
        ]
    out_sub = os.path.join(output_dir, "submissions")
    n_test = len(test)
    rows = []
    for name in expected_files:
        path = os.path.join(out_sub, name)
        rec = {"file": name, "path": path, "exists": os.path.exists(path)}
        if rec["exists"]:
            sub = pd.read_csv(path)
            rec["rows"] = len(sub)
            rec["rows_ok"] = len(sub) == n_test
            rec["cols_ok"] = list(sub.columns) == [id_col, "target"]
            rec["target_ok"] = sub["target"].between(0, 1).all() if "target" in sub.columns else False
            rec["id_ok"] = (sub[id_col] == test[id_col]).all() if id_col in sub.columns and id_col in test.columns else False
            rec["ok"] = rec["rows_ok"] and rec["cols_ok"] and rec["target_ok"] and rec["id_ok"]
        else:
            rec["rows"] = None
            rec["rows_ok"] = rec["cols_ok"] = rec["target_ok"] = rec["id_ok"] = rec["ok"] = False
        rows.append(rec)
    return pd.DataFrame(rows)


# 全設定名（エラー時やり直し用）
ALL_CONFIG_NAMES = [
    "baseline38", "v2", "v2_ratio", "v2_nmf", "v2_svd", "v2_all",
    "known_unknown", "v2_all_seed3",
]


def run_all_submission_configs(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: np.ndarray,
    baseline_features: list[str],
    output_dir: str,
    lgb_params: dict[str, Any] | None = None,
    val_years: list[int] | None = None,
    verbose: bool = True,
    time_aware: bool = False,
    submit_full_model: bool = True,
    skip_cv: bool = False,
    configs_to_run: list[str] | None = None,
    continue_on_error: bool = True,
) -> pd.DataFrame:
    """
    過去コンペの予測方法を複数パターンで実行し、提出ファイルを保存する。
    configs_to_run: 指定した設定名だけ実行。None なら全 8 種類。
    continue_on_error: True なら 1 つエラーしても続行し、結果に error 列で記録。
    戻り値: 各設定の config, n_features, cv_auc_mean, cv_auc_std, path, error の DataFrame。
    """
    out_sub = os.path.join(output_dir, "submissions")
    os.makedirs(out_sub, exist_ok=True)
    params = lgb_params or DEFAULT_LGB_PARAMS
    val_years = val_years or VAL_YEARS
    time_splits = None if skip_cv else get_time_splits(train, val_years)
    results = []
    run_set = set(configs_to_run) if configs_to_run is not None else set(ALL_CONFIG_NAMES)

    if not time_aware:
        global_mean, _, feature_sets = add_all_features(train, test, y, baseline_features)
    else:
        feature_sets = {"baseline": baseline_features}
        _, _, feature_sets = add_all_features(train, test, y, baseline_features)

    # 1. ベースライン38
    if "baseline38" in run_set:
        if verbose:
            print("=== ベースライン38 ===")
        try:
            if skip_cv:
                feats = feature_sets["baseline"]
                model_full = lgb.LGBMClassifier(**params)
                model_full.fit(train[feats], y)
                pred = model_full.predict_proba(test[feats])[:, 1]
                mean_auc, std_auc = np.nan, np.nan
            else:
                mean_auc, std_auc, test_preds = run_time_series_cv(
                    train, test, feature_sets["baseline"], y, time_splits, params, verbose=verbose
                )
                if submit_full_model:
                    feats = feature_sets["baseline"]
                    model_full = lgb.LGBMClassifier(**params)
                    model_full.fit(train[feats], y)
                    pred = model_full.predict_proba(test[feats])[:, 1]
                else:
                    pred = np.mean(test_preds, axis=0)
            path = os.path.join(out_sub, "submission_baseline38.csv")
            save_submission(test, pred, path)
            results.append({
                "config": "baseline38",
                "n_features": len(feature_sets["baseline"]),
                "cv_auc_mean": mean_auc,
                "cv_auc_std": std_auc,
                "path": path,
                "error": None,
            })
            if verbose:
                print(f"→ Saved {path}\n" if skip_cv else f"→ CV AUC: {mean_auc:.4f} ± {std_auc:.4f}, Saved {path}\n")
        except Exception as e:
            if continue_on_error:
                results.append({
                    "config": "baseline38", "n_features": None, "cv_auc_mean": np.nan, "cv_auc_std": np.nan,
                    "path": "", "error": str(e),
                })
                if verbose:
                    print(f"→ エラー: {e}\n")
            else:
                raise

    def _run_cv_v2(config_key: str, label: str, filename: str, verbose_label: str = ""):
        if config_key not in run_set:
            return
        if verbose:
            print(verbose_label or f"=== {config_key} ===")
        try:
            if skip_cv:
                feats = feature_sets[config_key]
                model_full = lgb.LGBMClassifier(**params)
                model_full.fit(train[feats], y)
                pred = model_full.predict_proba(test[feats])[:, 1]
                mean_auc, std_auc = np.nan, np.nan
            else:
                if time_aware:
                    mean_auc, std_auc, test_preds = run_time_series_cv_time_aware(
                        train, test, y, time_splits, baseline_features, config_key, params, verbose=verbose
                    )
                else:
                    mean_auc, std_auc, test_preds = run_time_series_cv(
                        train, test, feature_sets[config_key], y, time_splits, params, verbose=verbose
                    )
                if submit_full_model:
                    feats = feature_sets[config_key]
                    model_full = lgb.LGBMClassifier(**params)
                    model_full.fit(train[feats], y)
                    pred = model_full.predict_proba(test[feats])[:, 1]
                else:
                    pred = np.mean(test_preds, axis=0)
            path = os.path.join(out_sub, filename)
            save_submission(test, pred, path)
            nf = len(feature_sets[config_key]) if config_key in feature_sets else "-"
            results.append({"config": config_key, "n_features": nf, "cv_auc_mean": mean_auc, "cv_auc_std": std_auc, "path": path, "error": None})
            if verbose:
                print(f"→ Saved {path}\n" if skip_cv else f"→ CV AUC: {mean_auc:.4f} ± {std_auc:.4f}, Saved {path}\n")
        except Exception as e:
            if continue_on_error:
                results.append({
                    "config": config_key, "n_features": None, "cv_auc_mean": np.nan, "cv_auc_std": np.nan,
                    "path": "", "error": str(e),
                })
                if verbose:
                    print(f"→ エラー: {e}\n")
            else:
                raise

    _run_cv_v2("v2", "v2", "submission_v2.csv", "=== v2（OOF TE+カウント） ===")
    _run_cv_v2("v2_ratio", "v2_ratio", "submission_v2_ratio.csv", "=== v2 + 比率特徴量（1位・57位） ===")
    _run_cv_v2("v2_nmf", "v2_nmf", "submission_v2_nmf.csv", "=== v2 + NMF（3位: 批評家×映画） ===")
    _run_cv_v2("v2_svd", "v2_svd", "submission_v2_svd.csv", "=== v2 + CountVec+SVD（2位: id2vec風） ===")
    _run_cv_v2("v2_all", "v2_all", "submission_v2_all.csv", "=== v2 全入れ（比率+NMF+SVD） ===")

    # 7. 既知/未知 二モデル（1位・57位）
    if "known_unknown" in run_set:
        if verbose:
            print("=== 既知/未知 二モデル（1位・57位: seen/unseen別モデル） ===")
        try:
            if skip_cv:
                train_movies = set(train["rotten_tomatoes_link"])
                test["_is_known_movie"] = test["rotten_tomatoes_link"].isin(train_movies)
                feats_k = feature_sets["known"]
                feats_u = feature_sets["unknown"]
                model_k = lgb.LGBMClassifier(**params)
                model_k.fit(train[feats_k], y)
                model_u = lgb.LGBMClassifier(**params)
                model_u.fit(train[feats_u], y)
                pred = np.where(
                    test["_is_known_movie"].values,
                    model_k.predict_proba(test[feats_k])[:, 1],
                    model_u.predict_proba(test[feats_u])[:, 1],
                )
                test.drop(columns=["_is_known_movie"], inplace=True)
                summary = {}
            elif not time_aware:
                summary, pred = run_known_unknown_two_models(
                    train, test, y,
                    feature_sets["known"],
                    feature_sets["unknown"],
                    time_splits,
                    params,
                    verbose=verbose,
                )
                if submit_full_model:
                    train_movies = set(train["rotten_tomatoes_link"])
                    test["_is_known_movie"] = test["rotten_tomatoes_link"].isin(train_movies)
                    model_k = lgb.LGBMClassifier(**params)
                    model_k.fit(train[feature_sets["known"]], y)
                    model_u = lgb.LGBMClassifier(**params)
                    model_u.fit(train[feature_sets["unknown"]], y)
                    pred = np.where(
                        test["_is_known_movie"].values,
                        model_k.predict_proba(test[feature_sets["known"]])[:, 1],
                        model_u.predict_proba(test[feature_sets["unknown"]])[:, 1],
                    )
                    test.drop(columns=["_is_known_movie"], inplace=True)
            else:
                _, _, known_preds_ta = run_time_series_cv_time_aware(
                    train, test, y, time_splits, baseline_features, "v2_all", params, verbose=verbose
                )
                pred_known_avg = np.mean(known_preds_ta, axis=0)
                train_movies = set(train["rotten_tomatoes_link"])
                test["_is_known_movie"] = test["rotten_tomatoes_link"].isin(train_movies)
                _, _, feature_sets_ta = add_all_features(train, test, y, baseline_features)
                feats_unk = feature_sets_ta["unknown"]
                feats_known = feature_sets_ta["known"]
                if submit_full_model:
                    model_k = lgb.LGBMClassifier(**params)
                    model_k.fit(train[feats_known], y)
                    model_u = lgb.LGBMClassifier(**params)
                    model_u.fit(train[feats_unk], y)
                    pred = np.where(
                        test["_is_known_movie"].values,
                        model_k.predict_proba(test[feats_known])[:, 1],
                        model_u.predict_proba(test[feats_unk])[:, 1],
                    )
                else:
                    gkf = GroupKFold(n_splits=5)
                    groups_movie = train["rotten_tomatoes_link"].values
                    unknown_splits = list(gkf.split(train, y, groups=groups_movie))
                    X_unk = train[feats_unk]
                    X_unk_test = test[feats_unk]
                    preds_unk = []
                    for tr_idx, val_idx in unknown_splits:
                        model = lgb.LGBMClassifier(**params)
                        model.fit(X_unk.iloc[tr_idx], y[tr_idx], eval_set=[(X_unk.iloc[val_idx], y[val_idx])], callbacks=[lgb.early_stopping(30, verbose=False)])
                        preds_unk.append(model.predict_proba(X_unk_test)[:, 1])
                    pred_unknown_avg = np.mean(preds_unk, axis=0)
                    pred = np.where(test["_is_known_movie"].values, pred_known_avg, pred_unknown_avg)
                test.drop(columns=["_is_known_movie"], inplace=True)
                summary = {}
            path = os.path.join(out_sub, "submission_known_unknown.csv")
            save_submission(test, pred, path)
            results.append({
                "config": "known_unknown",
                "n_features": "known/unknown",
                "cv_auc_mean": summary.get("unknown_cv_auc", np.nan),
                "cv_auc_std": np.nan,
                "path": path,
                "error": None,
            })
            if verbose:
                print(f"→ Saved {path}\n")
        except Exception as e:
            if continue_on_error:
                results.append({
                    "config": "known_unknown", "n_features": None, "cv_auc_mean": np.nan, "cv_auc_std": np.nan,
                    "path": "", "error": str(e),
                })
                if verbose:
                    print(f"→ エラー: {e}\n")
            else:
                raise

    # 8. v2_all シードアベレージング（3位・57位）
    if "v2_all_seed3" in run_set:
        if verbose:
            print("=== v2_all シードアベレージング（3位・57位） ===")
        try:
            if skip_cv:
                feats = feature_sets["v2_all"]
                all_preds = []
                for seed in SEED_AVERAGING_SEEDS:
                    model = lgb.LGBMClassifier(**{**params, "random_state": seed})
                    model.fit(train[feats], y)
                    all_preds.append(model.predict_proba(test[feats])[:, 1])
                pred = np.mean(all_preds, axis=0)
                mean_auc, std_auc = np.nan, np.nan
            elif time_aware:
                all_means, all_preds_cv = [], []
                for seed in SEED_AVERAGING_SEEDS:
                    params_s = {**params, "random_state": seed}
                    mean_auc, std_auc, test_preds = run_time_series_cv_time_aware(
                        train, test, y, time_splits, baseline_features, "v2_all", params_s, verbose=False
                    )
                    all_means.append(mean_auc)
                    all_preds_cv.append(np.mean(test_preds, axis=0))
                mean_auc = float(np.mean(all_means))
                std_auc = float(np.std(all_means))
                if submit_full_model:
                    feats = feature_sets["v2_all"]
                    all_preds = []
                    for seed in SEED_AVERAGING_SEEDS:
                        model = lgb.LGBMClassifier(**{**params, "random_state": seed})
                        model.fit(train[feats], y)
                        all_preds.append(model.predict_proba(test[feats])[:, 1])
                    pred = np.mean(all_preds, axis=0)
                else:
                    pred = np.mean(all_preds_cv, axis=0)
            else:
                feats_v2_all = feature_sets["v2_all"]
                mean_auc, std_auc, pred = run_seed_averaging(
                    train, test, feats_v2_all, y, time_splits,
                    seeds=SEED_AVERAGING_SEEDS, lgb_params=params, verbose=verbose
                )
                if submit_full_model:
                    all_preds = []
                    for seed in SEED_AVERAGING_SEEDS:
                        model = lgb.LGBMClassifier(**{**params, "random_state": seed})
                        model.fit(train[feats_v2_all], y)
                        all_preds.append(model.predict_proba(test[feats_v2_all])[:, 1])
                    pred = np.mean(all_preds, axis=0)
            path = os.path.join(out_sub, "submission_v2_all_seed3.csv")
            save_submission(test, pred, path)
            nf_seed = len(feature_sets["v2_all"]) if "v2_all" in feature_sets else "-"
            results.append({
                "config": "v2_all_seed3",
                "n_features": nf_seed,
                "cv_auc_mean": mean_auc,
                "cv_auc_std": std_auc,
                "path": path,
                "error": None,
            })
            if verbose:
                print(f"→ Saved {path}\n" if skip_cv else f"→ CV AUC: {mean_auc:.4f} ± {std_auc:.4f}, Saved {path}\n")
        except Exception as e:
            if continue_on_error:
                results.append({
                    "config": "v2_all_seed3", "n_features": None, "cv_auc_mean": np.nan, "cv_auc_std": np.nan,
                    "path": "", "error": str(e),
                })
                if verbose:
                    print(f"→ エラー: {e}\n")
            else:
                raise

    return pd.DataFrame(results)


def retry_failed_submission_configs(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: np.ndarray,
    baseline_features: list[str],
    output_dir: str,
    summary_df: pd.DataFrame,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    前回 run_all_submission_configs でエラーになった設定だけをやり直す。
    summary_df に error 列がある場合は error が非 null の行、なければ path が存在しない行を失敗とみなす。
    戻り値: 元の summary_df の失敗行をやり直し結果で置き換えた DataFrame。
    """
    has_error = "error" in summary_df.columns
    if has_error:
        failed = summary_df[summary_df["error"].notna()]["config"].tolist()
    else:
        failed = [r["config"] for _, r in summary_df.iterrows() if not os.path.exists(r.get("path", ""))]
    if not failed:
        return summary_df
    retry_df = run_all_submission_configs(
        train, test, y, baseline_features, output_dir,
        configs_to_run=failed,
        continue_on_error=True,
        **kwargs,
    )
    # 失敗していた行をやり直し結果で置き換え
    out = summary_df.copy()
    for _, row in retry_df.iterrows():
        idx = out[out["config"] == row["config"]].index
        if len(idx):
            out.loc[idx[0]] = row
    if "error" not in out.columns:
        out["error"] = np.nan
    return out
