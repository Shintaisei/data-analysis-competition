#!/usr/bin/env python3
"""
ドキュメントで「まだ試せてない」案だけを、**次元削減は PCA 16 のみ**（重い ICA/UMAP 等は使わない）で提出ファイルを作成する。
処理が軽いので、残り提出回数でいろいろ試したいときに実行する。

参照: docs/01〜06, PROJECT_MIND — 追加実験（§4）、publisher_name_freq、シード平均、PCA24、アンサンブル

使い方:
  python run_quick_embedding_submissions.py
  # 既存 CSV は上書きしない（--skip-existing でスキップ）
  python run_quick_embedding_submissions.py --skip-existing

出力: outputs/submissions/ に 1 本ずつ保存される。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import lightgbm as lgb

from lib import get_baseline_data, BASELINE_FEATURES, BASELINE_LGB_PARAMS
from lib.embedding_reduction import fit_transform_embedding
from lib.submission import save_submission, verify_submission

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


def _run_one_and_save(train, test, y, base_features, E_train, E_test, n_comp, seed, out_path, skip_existing, tag):
    if skip_existing and out_path.exists():
        print(f"  [{tag}] スキップ（既存）")
        return True
    train_r, test_r, prefix = fit_transform_embedding(
        E_train, E_test, method="pca", n_components=n_comp, random_state=seed
    )
    red_names = [f"{prefix}_{i}" for i in range(train_r.shape[1])]
    feats = base_features + red_names
    X_tr = pd.concat([train[base_features].reset_index(drop=True), pd.DataFrame(train_r, columns=red_names)], axis=1)
    X_te = pd.concat([test[base_features].reset_index(drop=True), pd.DataFrame(test_r, columns=red_names)], axis=1)
    model = lgb.LGBMClassifier(**BASELINE_LGB_PARAMS)
    model.fit(X_tr[feats], y)
    pred = model.predict_proba(X_te[feats])[:, 1]
    save_submission(test, pred, out_path, sanitize=True)
    v = verify_submission(out_path, test)
    print(f"  [{tag}] → {out_path.name}  ({'OK' if v['ok'] else v['message']})")
    return True


def main():
    parser = argparse.ArgumentParser(description="PCA 16 のみでドキュメント未実施案の提出を一括作成")
    parser.add_argument("--skip-existing", action="store_true", help="既存 CSV はスキップ")
    args = parser.parse_args()

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    print("クイック提出（次元削減は PCA のみ）", flush=True)
    print(f"提出先: {SUBMISSIONS_DIR}", flush=True)

    train, test = get_baseline_data()
    y = train["target"].values
    base_features = list(BASELINE_FEATURES)
    safe = EMBEDDING_NAME.replace(" ", "_")

    # movie_title_info embedding（PCA 16 用）
    emb_df = load_embeddings(EMBEDDING_NAME)
    E_train = _merge_embeddings(train, emb_df)
    E_test = _merge_embeddings(test, emb_df)

    saved = []

    # --- 1. seed_avg: シード 42,43,44 で 3 本 → 予測を平均 ---
    fname_seed = f"submission_embedding_{safe}_pca{N_COMPONENTS}_seed_avg.csv"
    path_seed = SUBMISSIONS_DIR / fname_seed
    if not args.skip_existing or not path_seed.exists():
        preds = []
        for rs in [42, 43, 44]:
            train_r, test_r, prefix = fit_transform_embedding(
                E_train, E_test, method="pca", n_components=N_COMPONENTS, random_state=rs
            )
            red_names = [f"{prefix}_{i}" for i in range(train_r.shape[1])]
            feats = base_features + red_names
            X_tr = pd.concat([train[base_features].reset_index(drop=True), pd.DataFrame(train_r, columns=red_names)], axis=1)
            X_te = pd.concat([test[base_features].reset_index(drop=True), pd.DataFrame(test_r, columns=red_names)], axis=1)
            model = lgb.LGBMClassifier(**BASELINE_LGB_PARAMS)
            model.fit(X_tr[feats], y)
            preds.append(model.predict_proba(X_te[feats])[:, 1])
        pred_avg = np.mean(preds, axis=0)
        save_submission(test, pred_avg, path_seed, sanitize=True)
        v = verify_submission(path_seed, test)
        print(f"  [seed_avg] → {fname_seed}  ({'OK' if v['ok'] else v['message']})")
    else:
        print(f"  [seed_avg] スキップ（既存）")
    saved.append(fname_seed)

    # --- 2. pca24: PCA 24 次元 ---
    fname_24 = f"submission_embedding_{safe}_pca24.csv"
    path_24 = SUBMISSIONS_DIR / fname_24
    _run_one_and_save(train, test, y, base_features, E_train, E_test, 24, 42, path_24, args.skip_existing, "pca24")
    saved.append(fname_24)

    # --- 3. ensemble_two: movie_title_info_pca16 + movie_title_info_large_pca16 の 0.5:0.5 ---
    # large の CSV が無ければ 1 本だけ作成する
    p1 = SUBMISSIONS_DIR / "submission_embedding_movie_title_info_pca16.csv"
    p2 = SUBMISSIONS_DIR / "submission_embedding_movie_title_info_large_pca16.csv"
    if not p2.exists():
        print("  [large_pca16] movie_title_info_large + PCA 16 を 1 本作成（ensemble_two 用）...", flush=True)
        try:
            emb_large = load_embeddings("movie_title_info_large")
            E_tr_l = _merge_embeddings(train, emb_large)
            E_te_l = _merge_embeddings(test, emb_large)
            _run_one_and_save(train, test, y, base_features, E_tr_l, E_te_l, N_COMPONENTS, 42, p2, False, "large_pca16")
        except FileNotFoundError as e:
            print(f"  [ensemble_two] スキップ: {e}")
    fname_ens = "submission_embedding_ensemble_movie_title_info_and_large_pca16.csv"
    path_ens = SUBMISSIONS_DIR / fname_ens
    if (not args.skip_existing or not path_ens.exists()) and p1.exists() and p2.exists():
        d1 = pd.read_csv(p1)
        d2 = pd.read_csv(p2)
        merged = d1[["ID", "target"]].merge(d2[["ID", "target"]], on="ID", suffixes=("_1", "_2"))
        merged["target"] = (merged["target_1"].astype(float) + merged["target_2"].astype(float)) / 2
        test_pred = test[["ID"]].merge(merged[["ID", "target"]], on="ID", how="left")
        if test_pred["target"].isna().any():
            print("  [ensemble_two] スキップ: ID 不一致")
        else:
            save_submission(test, test_pred["target"].values, path_ens, sanitize=True)
            v = verify_submission(path_ens, test)
            print(f"  [ensemble_two] → {fname_ens}  ({'OK' if v['ok'] else v['message']})")
        saved.append(fname_ens)
    elif args.skip_existing and path_ens.exists():
        print(f"  [ensemble_two] スキップ（既存）")
        saved.append(fname_ens)

    # --- 4. publisher_freq: ベースライン + embedding PCA16 + publisher_name 出現回数（01_PREPROCESS で +0.0003） ---
    if "publisher_name" in train.columns:
        pub_count = train["publisher_name"].value_counts()
        train = train.copy()
        test = test.copy()
        train["publisher_name_freq"] = train["publisher_name"].map(pub_count).fillna(0).astype(np.int32)
        test["publisher_name_freq"] = test["publisher_name"].map(pub_count).fillna(0).astype(np.int32)
        base_plus_freq = base_features + ["publisher_name_freq"]
        fname_pub = f"submission_embedding_{safe}_pca{N_COMPONENTS}_publisher_freq.csv"
        path_pub = SUBMISSIONS_DIR / fname_pub
        if not args.skip_existing or not path_pub.exists():
            train_r, test_r, prefix = fit_transform_embedding(
                E_train, E_test, method="pca", n_components=N_COMPONENTS, random_state=42
            )
            red_names = [f"{prefix}_{i}" for i in range(train_r.shape[1])]
            feats = base_plus_freq + red_names
            X_tr = pd.concat([
                train[base_plus_freq].reset_index(drop=True),
                pd.DataFrame(train_r, columns=red_names),
            ], axis=1)
            X_te = pd.concat([
                test[base_plus_freq].reset_index(drop=True),
                pd.DataFrame(test_r, columns=red_names),
            ], axis=1)
            model = lgb.LGBMClassifier(**BASELINE_LGB_PARAMS)
            model.fit(X_tr[feats], y)
            pred = model.predict_proba(X_te[feats])[:, 1]
            save_submission(test, pred, path_pub, sanitize=True)
            v = verify_submission(path_pub, test)
            print(f"  [publisher_freq] → {fname_pub}  ({'OK' if v['ok'] else v['message']})")
        else:
            print(f"  [publisher_freq] スキップ（既存）")
        saved.append(fname_pub)
    else:
        print("  [publisher_freq] スキップ: publisher_name なし")

    print(f"\n保存先: {SUBMISSIONS_DIR}")
    print(f"対象: {len(saved)} 件")


if __name__ == "__main__":
    main()
