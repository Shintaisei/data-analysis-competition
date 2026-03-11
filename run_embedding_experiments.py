#!/usr/bin/env python3
"""
Embedding 実験を一括実行し、結果を CSV に保存する。
バックグラウンド実行: nohup python run_embedding_experiments.py > outputs/embedding_experiments.log 2>&1 &

- 4 種類の embedding: movie_info, movie_info_large, movie_title_info, movie_title_info_large
- 各 2 パターン: PCA 8 次元, PCA 16 次元
- 検証: (1) 時系列CV 2013–2016, (2) GroupKFold by movie（ベースラインv2 風）
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

# プロジェクトルートを path に追加
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from lib import get_baseline_data, BASELINE_FEATURES, BASELINE_LGB_PARAMS

OUTPUT_DIR = ROOT / "outputs"
EMBEDDINGS_DIR = OUTPUT_DIR / "embeddings"
RESULTS_PATH = OUTPUT_DIR / "embedding_experiment_results.csv"

# 4 種類の embedding ファイル（それぞれ 2 パターン: PCA 8, PCA 16）
EMBEDDING_CONFIGS = [
    {"name": "movie_info", "path": EMBEDDINGS_DIR / "movie_info_embeddings.pkl", "loader": "movie_info"},
    {"name": "movie_info_large", "path": EMBEDDINGS_DIR / "movie_info_embeddings_large.pkl", "loader": "movie_info"},
    {"name": "movie_title_info", "path": EMBEDDINGS_DIR / "movie_title_info_embeddings.pkl", "loader": "title_info"},
    {"name": "movie_title_info_large", "path": EMBEDDINGS_DIR / "movie_title_info_embeddings_large.pkl", "loader": "title_info"},
]
PCA_PATTERNS = [8, 16]  # 2 パターンずつ
VAL_YEARS = [2013, 2014, 2015, 2016]
GROUP_KFOLD_N = 5


def load_embeddings(config: dict) -> pd.DataFrame:
    """loader に応じて movie_info または title_info を読み込む。"""
    path = config["path"]
    if not path.exists():
        return None
    loader = config["loader"]
    if loader == "movie_info":
        from lib.openai_embeddings import load_movie_info_embeddings
        return load_movie_info_embeddings(path=path)
    if loader == "title_info":
        from lib.openai_embeddings import load_movie_title_info_embeddings
        return load_movie_title_info_embeddings(path=path)
    return None


def _merge_embeddings(
    df: pd.DataFrame,
    emb_df: pd.DataFrame,
) -> np.ndarray:
    """df に embedding を left join し、emb 行列 (n_rows, n_emb_dim) を返す。"""
    emb_cols = [c for c in emb_df.columns if c != "rotten_tomatoes_link"]
    merged = df[["rotten_tomatoes_link"]].merge(
        emb_df[["rotten_tomatoes_link"] + emb_cols],
        on="rotten_tomatoes_link",
        how="left",
    )
    return merged[emb_cols].fillna(0).astype(np.float32).values


def add_embedding_pca(
    train: pd.DataFrame,
    test: pd.DataFrame,
    emb_df: pd.DataFrame,
    n_components: int,
    prefix: str = "emb_pca",
    random_state: int = 42,
) -> list[str]:
    """
    Embedding を PCA 圧縮して train/test に列を追加。
    注意: この関数は「全 train」で PCA fit するため、CV 内で使う場合は fold 内 tr だけで fit する
    run_*_cv の fold 内で _fit_pca_on_tr_and_transform を使う実装にしてある。
    """
    emb_cols = [c for c in emb_df.columns if c != "rotten_tomatoes_link"]
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
    n_comp = min(n_components, len(emb_cols), X_tr.shape[1])
    pca = PCA(n_components=n_comp, random_state=random_state)
    tr_pca = pca.fit_transform(X_tr)
    te_pca = pca.transform(X_te)
    names = [f"{prefix}_{i}" for i in range(n_comp)]
    for i, name in enumerate(names):
        train[name] = tr_pca[:, i]
        test[name] = te_pca[:, i]
    return names


def get_time_splits(train: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    """時系列CV: 各 val_year より前で学習・その年で検証。"""
    splits = []
    for vy in VAL_YEARS:
        tr_idx = np.where(train["review_year"] < vy)[0]
        val_idx = np.where(train["review_year"] == vy)[0]
        if len(val_idx) > 0:
            splits.append((tr_idx, val_idx))
    return splits


def run_time_series_cv(
    train: pd.DataFrame,
    test: pd.DataFrame,
    base_features: list[str],
    y: np.ndarray,
    time_splits: list[tuple[np.ndarray, np.ndarray]],
    emb_df: pd.DataFrame,
    n_comp: int,
    prefix: str,
) -> tuple[float, float]:
    """
    時系列CV: 各 fold で「学習データ（tr_idx）だけで PCA を fit」し、val/test はその PCA で transform。
    学習 = review_year < val_year, 検証 = review_year == val_year（時系列リークなし）。
    """
    E_train = _merge_embeddings(train, emb_df)
    E_test = _merge_embeddings(test, emb_df)
    n_comp = min(n_comp, E_train.shape[1])
    pca_names = [f"{prefix}_{i}" for i in range(n_comp)]
    fold_scores = []
    for tr_idx, val_idx in time_splits:
        pca = PCA(n_components=n_comp, random_state=42).fit(E_train[tr_idx])
        train_pca = pca.transform(E_train)
        X_tr = pd.concat([
            train[base_features].iloc[tr_idx].reset_index(drop=True),
            pd.DataFrame(train_pca[tr_idx], columns=pca_names),
        ], axis=1)
        X_val = pd.concat([
            train[base_features].iloc[val_idx].reset_index(drop=True),
            pd.DataFrame(train_pca[val_idx], columns=pca_names),
        ], axis=1)
        y_tr, y_val = y[tr_idx], y[val_idx]
        model = lgb.LGBMClassifier(**BASELINE_LGB_PARAMS)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
        fold_scores.append(auc)
    return float(np.mean(fold_scores)), float(np.std(fold_scores))


def run_group_kfold_cv(
    train: pd.DataFrame,
    test: pd.DataFrame,
    base_features: list[str],
    y: np.ndarray,
    group_splits: list[tuple[np.ndarray, np.ndarray]],
    emb_df: pd.DataFrame,
    n_comp: int,
    prefix: str,
) -> tuple[float, float]:
    """
    GroupKFold（group=映画）: 各 fold で「学習データ（tr_idx）だけで PCA を fit」し、val はその PCA で transform。
    同じ映画は tr か val のどちらか一方のみ（映画単位でリークなし）。
    """
    E_train = _merge_embeddings(train, emb_df)
    n_comp = min(n_comp, E_train.shape[1])
    pca_names = [f"{prefix}_{i}" for i in range(n_comp)]
    fold_scores = []
    for tr_idx, val_idx in group_splits:
        pca = PCA(n_components=n_comp, random_state=42).fit(E_train[tr_idx])
        train_pca = pca.transform(E_train)
        X_tr = pd.concat([
            train[base_features].iloc[tr_idx].reset_index(drop=True),
            pd.DataFrame(train_pca[tr_idx], columns=pca_names),
        ], axis=1)
        X_val = pd.concat([
            train[base_features].iloc[val_idx].reset_index(drop=True),
            pd.DataFrame(train_pca[val_idx], columns=pca_names),
        ], axis=1)
        y_tr, y_val = y[tr_idx], y[val_idx]
        model = lgb.LGBMClassifier(**BASELINE_LGB_PARAMS)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
        fold_scores.append(auc)
    return float(np.mean(fold_scores)), float(np.std(fold_scores))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("データ読み込み...", flush=True)
    train, test = get_baseline_data()
    y = train["target"].values
    time_splits = get_time_splits(train)
    groups = train["rotten_tomatoes_link"].values
    gkf = GroupKFold(n_splits=GROUP_KFOLD_N)
    group_splits = list(gkf.split(train, y, groups=groups))
    print(f"  train={len(train):,}, test={len(test):,}, 時系列 folds={len(time_splits)}, GroupKFold n={GROUP_KFOLD_N}", flush=True)

    base_features = list(BASELINE_FEATURES)
    results = []

    for config in EMBEDDING_CONFIGS:
        emb_df = load_embeddings(config)
        if emb_df is None:
            print(f"  [SKIP] {config['name']}: ファイルなし {config['path']}", flush=True)
            results.append({
                "embedding": config["name"],
                "pattern": "PCA_8",
                "time_cv_auc_mean": np.nan,
                "time_cv_auc_std": np.nan,
                "group_kfold_auc_mean": np.nan,
                "group_kfold_auc_std": np.nan,
            })
            results.append({
                "embedding": config["name"],
                "pattern": "PCA_16",
                "time_cv_auc_mean": np.nan,
                "time_cv_auc_std": np.nan,
                "group_kfold_auc_mean": np.nan,
                "group_kfold_auc_std": np.nan,
            })
            continue

        for n_comp in PCA_PATTERNS:
            prefix = f"emb_{config['name'][:8]}_{n_comp}"
            print(f"  [{config['name']}] PCA {n_comp} ...", flush=True)
            ts_mean, ts_std = run_time_series_cv(
                train, test, base_features, y, time_splits, emb_df, n_comp, prefix
            )
            gk_mean, gk_std = run_group_kfold_cv(
                train, test, base_features, y, group_splits, emb_df, n_comp, prefix
            )
            print(f"    時系列CV AUC: {ts_mean:.4f} ± {ts_std:.4f}, GroupKFold AUC: {gk_mean:.4f} ± {gk_std:.4f}", flush=True)
            results.append({
                "embedding": config["name"],
                "pattern": f"PCA_{n_comp}",
                "time_cv_auc_mean": ts_mean,
                "time_cv_auc_std": ts_std,
                "group_kfold_auc_mean": gk_mean,
                "group_kfold_auc_std": gk_std,
            })

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_PATH, index=False)
    print(f"\n結果を保存しました: {RESULTS_PATH}", flush=True)
    print(df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
