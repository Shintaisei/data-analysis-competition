"""
改善10候補の実行ロジック。train_improvement_10_candidates.ipynb から呼び出す。

前提（デフォルト）:
- 土台は 0.75493 を出したパイプライン: get_baseline_data + movie_title_info embedding + PCA 16
  + doc_x_critic_te（critic_te × genre_Documentary）1 列。特徴数は 38 + 16 + 1 = 55。
- use_best_pipeline=False のときのみ 38 特徴のみ（従来どおり）。

スコア更新の狙い:
- 改善候補（01, 05, 06, 07, 08, 10 など）はこの 55 特徴の上で動くので、0.75493 を超える提出を狙える。
- 04 ブレンドは既存 CSV を混ぜるだけなので従来どおり。
"""
from __future__ import annotations

import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

import lightgbm as lgb

from .embedding_reduction import fit_transform_embedding
from .pipeline import get_baseline_data, BASELINE_FEATURES, BASELINE_LGB_PARAMS
from .submission import save_submission, verify_submission

VAL_YEARS = [2013, 2014, 2015, 2016]

# 0.75493 を出したパイプライン（quick_embedding の doc_x_critic_te）と同じ設定
EMBEDDING_NAME = "movie_title_info"
EMBEDDING_N_COMPONENTS = 16
DOC_FEATURE = "critic_te_x_genre_Documentary"
EMBEDDINGS_DIR = Path("outputs") / "embeddings"
EMBEDDING_CONFIGS = {
    "movie_info": {"path": EMBEDDINGS_DIR / "movie_info_embeddings.pkl", "loader": "movie_info"},
    "movie_title_info": {"path": EMBEDDINGS_DIR / "movie_title_info_embeddings.pkl", "loader": "title_info"},
    "movie_title_info_large": {"path": EMBEDDINGS_DIR / "movie_title_info_embeddings_large.pkl", "loader": "title_info"},
}


class ImprovementContext(NamedTuple):
    """改善候補実行用の共通コンテキスト。"""
    train: pd.DataFrame
    test: pd.DataFrame
    X: pd.DataFrame
    X_test: pd.DataFrame
    y: np.ndarray
    features: list[str]
    time_splits: list[tuple[np.ndarray, np.ndarray]]
    submissions_dir: Path
    lgb_params: dict[str, Any]


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def _load_embedding_merged(
    train: pd.DataFrame,
    test: pd.DataFrame,
    embeddings_dir: Path,
    name: str = EMBEDDING_NAME,
) -> tuple[np.ndarray, np.ndarray]:
    """embedding を読み、train/test にマージした行列 (E_train, E_test) を返す。"""
    if name not in EMBEDDING_CONFIGS:
        raise ValueError(f"不明な embedding: {name}. 利用可能: {list(EMBEDDING_CONFIGS.keys())}")
    config = EMBEDDING_CONFIGS[name]
    path = embeddings_dir / Path(config["path"]).name
    if not path.exists():
        path = path.with_suffix(".parquet")
    if not path.exists():
        raise FileNotFoundError(
            f"embedding がありません: {path}. "
            "quick_embedding_submissions.ipynb または run_openai_embeddings で movie_title_info を生成してください。"
        )
    if config.get("loader") == "movie_info":
        from .openai_embeddings import load_movie_info_embeddings
        emb_df = load_movie_info_embeddings(path=path)
    else:
        from .openai_embeddings import load_movie_title_info_embeddings
        emb_df = load_movie_title_info_embeddings(path=path)
    emb_cols = [c for c in emb_df.columns if c != "rotten_tomatoes_link"]
    m_tr = train[["rotten_tomatoes_link"]].merge(
        emb_df[["rotten_tomatoes_link"] + emb_cols], on="rotten_tomatoes_link", how="left"
    )
    m_te = test[["rotten_tomatoes_link"]].merge(
        emb_df[["rotten_tomatoes_link"] + emb_cols], on="rotten_tomatoes_link", how="left"
    )
    E_train = m_tr[emb_cols].fillna(0).astype(np.float32).values
    E_test = m_te[emb_cols].fillna(0).astype(np.float32).values
    return E_train, E_test


def _build_best_pipeline_data(
    seed: int,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], np.ndarray]:
    """0.75493 パイプラインの train/test と特徴名リスト・y を返す。"""
    train, test = get_baseline_data()
    train = train.copy()
    test = test.copy()
    train[DOC_FEATURE] = train["critic_name_te_ts"].astype(float) * train["genre_Documentary"].astype(float)
    test[DOC_FEATURE] = test["critic_name_te_ts"].astype(float) * test["genre_Documentary"].astype(float)

    embeddings_dir = output_dir / "embeddings"
    E_train, E_test = _load_embedding_merged(train, test, embeddings_dir, EMBEDDING_NAME)
    train_r, test_r, prefix = fit_transform_embedding(
        E_train, E_test, method="pca", n_components=EMBEDDING_N_COMPONENTS, random_state=seed
    )
    red_names = [f"{prefix}_{i}" for i in range(train_r.shape[1])]
    for i, c in enumerate(red_names):
        train[c] = train_r[:, i]
        test[c] = test_r[:, i]

    features = list(BASELINE_FEATURES) + red_names + [DOC_FEATURE]
    y = train["target"].values
    return train, test, features, y


def get_setup(
    seed: int = 42,
    output_dir: str | Path = "outputs",
    use_best_pipeline: bool = True,
) -> ImprovementContext:
    """データ読み込み・時系列CV分割・共通変数を用意する。
    use_best_pipeline=True（デフォルト）のとき、0.75493 の土台（embedding + PCA16 + doc_x_critic_te）を使う。"""
    seed_everything(seed)
    out = Path(output_dir)
    submissions_dir = out / "submissions"
    submissions_dir.mkdir(parents=True, exist_ok=True)

    if use_best_pipeline:
        train, test, features, y = _build_best_pipeline_data(seed, out)
        X = train[features]
        X_test = test[features]
    else:
        train, test = get_baseline_data()
        features = list(BASELINE_FEATURES)
        y = train["target"].values
        X = train[features]
        X_test = test[features]

    time_splits = []
    for vy in VAL_YEARS:
        tr_idx = np.where(train["review_year"] < vy)[0]
        val_idx = np.where(train["review_year"] == vy)[0]
        if len(val_idx) > 0:
            time_splits.append((tr_idx, val_idx))

    return ImprovementContext(
        train=train,
        test=test,
        X=X,
        X_test=X_test,
        y=y,
        features=features,
        time_splits=time_splits,
        submissions_dir=submissions_dir,
        lgb_params=dict(BASELINE_LGB_PARAMS),
    )


def _save_and_verify(
    test: pd.DataFrame,
    pred: np.ndarray,
    path: Path,
) -> dict[str, Any]:
    save_submission(test, pred, path, sanitize=True)
    return verify_submission(path, test)


def run_05_scale_pos_weight(ctx: ImprovementContext) -> dict[str, Any]:
    """Rotten 重視。scale_pos_weight = n_neg/n_pos で positive(Fresh) を相対的に減らし、Rotten を重視。"""
    n_pos = int(ctx.y.sum())
    n_neg = len(ctx.y) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)
    params = {**ctx.lgb_params, "scale_pos_weight": scale_pos_weight}
    model = lgb.LGBMClassifier(**params)
    model.fit(ctx.X, ctx.y)
    pred = model.predict_proba(ctx.X_test)[:, 1]
    path = ctx.submissions_dir / "submission_improvement_05_scale_pos_weight.csv"
    return _save_and_verify(ctx.test, pred, path)


def run_10_extra_col(ctx: ImprovementContext) -> dict[str, Any]:
    """弱点1列追加（year_norm_x_critic_te）。"""
    train_10 = ctx.train.copy()
    test_10 = ctx.test.copy()
    ry_min = ctx.train["review_year"].min()
    ry_max = ctx.train["review_year"].max()
    denom = max(ry_max - ry_min, 1)
    train_10["review_year_norm"] = (ctx.train["review_year"] - ry_min) / denom
    test_10["review_year_norm"] = (ctx.test["review_year"] - ry_min) / denom
    train_10["year_norm_x_critic_te"] = (
        train_10["review_year_norm"].astype(float) * train_10["critic_name_te_ts"].astype(float)
    )
    test_10["year_norm_x_critic_te"] = (
        test_10["review_year_norm"].astype(float) * test_10["critic_name_te_ts"].astype(float)
    )
    feats = ctx.features + ["year_norm_x_critic_te"]
    X_tr = train_10[feats]
    X_te = test_10[feats]
    model = lgb.LGBMClassifier(**ctx.lgb_params)
    model.fit(X_tr, ctx.y)
    pred = model.predict_proba(X_te)[:, 1]
    path = ctx.submissions_dir / "submission_improvement_10_extra_col.csv"
    return _save_and_verify(ctx.test, pred, path)


def run_07_feature_selection(ctx: ImprovementContext) -> dict[str, Any]:
    """全foldで重要度0の列を除去して再学習。"""
    importance_by_fold = {f: [] for f in ctx.features}
    for tr_idx, val_idx in ctx.time_splits:
        X_tr = ctx.X.iloc[tr_idx]
        X_val = ctx.X.iloc[val_idx]
        y_tr = ctx.y[tr_idx]
        y_val = ctx.y[val_idx]
        m = lgb.LGBMClassifier(**ctx.lgb_params)
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(30, verbose=False)])
        for feat, imp in zip(ctx.features, m.feature_importances_):
            importance_by_fold[feat].append(imp)

    zero_importance = [f for f in ctx.features if sum(importance_by_fold[f]) == 0]
    feats_07 = [f for f in ctx.features if f not in zero_importance]

    X_07 = ctx.train[feats_07]
    X_test_07 = ctx.test[feats_07]
    model = lgb.LGBMClassifier(**ctx.lgb_params)
    model.fit(X_07, ctx.y)
    pred = model.predict_proba(X_test_07)[:, 1]
    path = ctx.submissions_dir / "submission_improvement_07_feature_selection.csv"
    result = _save_and_verify(ctx.test, pred, path)
    result["n_dropped"] = len(zero_importance)
    result["n_kept"] = len(feats_07)
    return result


def run_06_groupkfold(ctx: ImprovementContext, n_splits: int = 4) -> dict[str, Any]:
    """GroupKFold で学習し、fold 予測の平均を提出。
    注意: 4本とも「全データではなく約75%で学習」したモデルの平均のため、
    38特徴・全データ1本のベースライン(0.75097)より Public が下がる設計。"""
    groups = ctx.train["rotten_tomatoes_link"].astype(str)
    gkf = GroupKFold(n_splits=n_splits)
    test_preds = []
    fold_scores = []
    for tr_idx, val_idx in gkf.split(ctx.X, ctx.y, groups):
        X_tr = ctx.X.iloc[tr_idx]
        X_val = ctx.X.iloc[val_idx]
        y_tr = ctx.y[tr_idx]
        y_val = ctx.y[val_idx]
        m = lgb.LGBMClassifier(**ctx.lgb_params)
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(30, verbose=False)])
        fold_scores.append(roc_auc_score(y_val, m.predict_proba(X_val)[:, 1]))
        test_preds.append(m.predict_proba(ctx.X_test)[:, 1])

    pred = np.mean(test_preds, axis=0)
    path = ctx.submissions_dir / "submission_improvement_06_groupkfold.csv"
    result = _save_and_verify(ctx.test, pred, path)
    result["gkf_auc_mean"] = float(np.mean(fold_scores))
    return result


def run_01_pseudo_label(
    ctx: ImprovementContext,
    high_thresh: float = 0.95,
    low_thresh: float = 0.05,
    max_pseudo: int | None = 1500,
    pseudo_weight: float = 0.3,
) -> dict[str, Any]:
    """高確信度 test を擬似ラベルにして再学習。max_pseudo で件数キャップ、pseudo_weight で重みを下げて過学習を抑制。"""
    model_base = lgb.LGBMClassifier(**ctx.lgb_params)
    model_base.fit(ctx.X, ctx.y)
    test_proba = model_base.predict_proba(ctx.X_test)[:, 1]

    high = test_proba >= high_thresh
    low = test_proba <= low_thresh
    pseudo_idx = np.where(high | low)[0]

    if len(pseudo_idx) == 0:
        pred = test_proba
        n_pseudo = 0
    else:
        # 確信度が高い順に並べ、max_pseudo 件だけ使う（ノイズ量をキャップ）
        confidence = np.where(
            test_proba[pseudo_idx] >= 0.5,
            test_proba[pseudo_idx],
            1.0 - test_proba[pseudo_idx],
        )
        order = np.argsort(-confidence)
        if max_pseudo is not None and len(pseudo_idx) > max_pseudo:
            order = order[:max_pseudo]
        pseudo_idx = pseudo_idx[order]

        X_pseudo = ctx.test.iloc[pseudo_idx][ctx.features].copy()
        y_pseudo = np.where(test_proba[pseudo_idx] >= 0.5, 1, 0).astype(np.float64)
        X_agg = pd.concat([ctx.X, X_pseudo], axis=0, ignore_index=True)
        for col in X_agg.columns:
            if not pd.api.types.is_numeric_dtype(X_agg[col]) and not pd.api.types.is_categorical_dtype(X_agg[col]):
                X_agg[col] = X_agg[col].astype("category")
        y_agg = np.concatenate([ctx.y, y_pseudo])
        # 実ラベル重み 1.0、擬似ラベルは pseudo_weight で過学習を抑える
        sample_weight = np.concatenate([
            np.ones(len(ctx.y)),
            np.full(len(pseudo_idx), pseudo_weight),
        ])
        model_01 = lgb.LGBMClassifier(**ctx.lgb_params)
        model_01.fit(X_agg, y_agg, sample_weight=sample_weight)
        pred = model_01.predict_proba(ctx.X_test)[:, 1]
        n_pseudo = len(pseudo_idx)

    path = ctx.submissions_dir / "submission_improvement_01_pseudo_label.csv"
    result = _save_and_verify(ctx.test, pred, path)
    result["n_pseudo"] = n_pseudo
    return result


def run_03_stacking(ctx: ImprovementContext, seed: int | None = 42) -> dict[str, Any]:
    """LGB + XGB + CatBoost → Ridge スタッキング。seed で乱数固定（複数 seed で出し分け可能）。"""
    try:
        import xgboost as xgb
        import catboost as cb
    except ImportError as e:
        return {"ok": False, "message": f"xgboost/catboost 未導入: {e}"}

    use_seed = seed if seed is not None else 42
    xgb_params = {"objective": "binary:logistic", "eval_metric": "auc", "random_state": use_seed, "verbosity": 0}
    cb_params = {"iterations": 1000, "learning_rate": 0.05, "verbose": False, "random_seed": use_seed}
    lgb_params = {**ctx.lgb_params, "random_state": use_seed}

    test_meta_list = []
    for tr_idx, val_idx in ctx.time_splits:
        X_tr = ctx.X.iloc[tr_idx]
        X_val = ctx.X.iloc[val_idx]
        y_tr = ctx.y[tr_idx]
        y_val = ctx.y[val_idx]

        ml = lgb.LGBMClassifier(**lgb_params)
        ml.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(30, verbose=False)])
        oof_lgb = ml.predict_proba(X_val)[:, 1]
        t_lgb = ml.predict_proba(ctx.X_test)[:, 1]

        mx = xgb.XGBClassifier(**xgb_params, early_stopping_rounds=30)
        mx.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        oof_xgb = mx.predict_proba(X_val)[:, 1]
        t_xgb = mx.predict_proba(ctx.X_test)[:, 1]

        mc = cb.CatBoostClassifier(**cb_params, early_stopping_rounds=30)
        mc.fit(X_tr, y_tr, eval_set=(X_val, y_val))
        oof_cb = mc.predict_proba(X_val)[:, 1]
        t_cb = mc.predict_proba(ctx.X_test)[:, 1]

        meta = np.column_stack([oof_lgb, oof_xgb, oof_cb])
        ridge = Ridge(alpha=1.0, random_state=use_seed)
        ridge.fit(meta, y_val)
        test_meta_list.append(ridge.predict(np.column_stack([t_lgb, t_xgb, t_cb])))

    pred = np.mean(test_meta_list, axis=0)
    fname = "submission_improvement_03_stacking.csv" if seed is None else f"submission_improvement_03_stacking_seed{use_seed}.csv"
    path = ctx.submissions_dir / fname
    return _save_and_verify(ctx.test, pred, path)


def run_03_stacking_batch(
    ctx: ImprovementContext,
    seeds: list[int] | None = None,
) -> list[dict[str, Any]]:
    """03 スタッキングを複数 seed で実行し、提出ファイルを 1 本ずつ保存。デフォルトで 5 本（seed 42〜46）。"""
    if seeds is None:
        seeds = [42, 43, 44, 45, 46]
    results = []
    for s in seeds:
        r = run_03_stacking(ctx, seed=s)
        results.append(r)
        if r.get("path"):
            print(f"  [03 stacking seed{s}] → {r['path'].name}  ({'OK' if r.get('ok') else r.get('message', '')})")
        else:
            print(f"  [03 stacking seed{s}] スキップ: {r.get('message', '')}")
    return results


def run_08_tfidf_svd(
    ctx: ImprovementContext,
    max_features: int = 200,
    n_components: int = 20,
) -> dict[str, Any]:
    """movie_info の TF-IDF → SVD を追加特徴として学習。"""
    tfidf = TfidfVectorizer(
        max_features=max_features,
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    svd = TruncatedSVD(n_components=n_components, random_state=42)

    # Categorical の列は fillna("") が使えないため、先に str に変換してから欠損を空文字に
    text_tr = ctx.train["movie_info"].astype(str).replace("nan", "")
    text_te = ctx.test["movie_info"].astype(str).replace("nan", "")
    M_tr = svd.fit_transform(tfidf.fit_transform(text_tr))
    M_te = svd.transform(tfidf.transform(text_te))

    tfidf_cols = [f"tfidf_svd_{i}" for i in range(n_components)]
    train_08 = ctx.train.copy()
    test_08 = ctx.test.copy()
    for i, c in enumerate(tfidf_cols):
        train_08[c] = M_tr[:, i]
        test_08[c] = M_te[:, i]

    feats = ctx.features + tfidf_cols
    X_tr = train_08[feats]
    X_te = test_08[feats]

    model = lgb.LGBMClassifier(**ctx.lgb_params)
    model.fit(X_tr, ctx.y)
    pred = model.predict_proba(X_te)[:, 1]
    path = ctx.submissions_dir / "submission_improvement_08_tfidf_svd.csv"
    return _save_and_verify(ctx.test, pred, path)


def run_04_blend(
    ctx: ImprovementContext,
    blend_files: list[str] | None = None,
) -> dict[str, Any]:
    """既存提出 CSV をブレンド。"""
    if blend_files is None:
        blend_files = [
            "submission_embedding_movie_title_info_pca16_doc_x_critic_te.csv",  # 最高 0.75493
            "submission_embedding_movie_title_info_pca8.csv",
            "submission_embedding_movie_title_info_pca16.csv",
            "submission_embedding_movie_title_info_pca32.csv",
            "submission.csv",
        ]
    dfs = []
    for f in blend_files:
        p = ctx.submissions_dir / f
        if p.exists():
            dfs.append(pd.read_csv(p)[["ID", "target"]].rename(columns={"target": f}))

    if len(dfs) < 2:
        return {"ok": False, "message": f"ブレンド用CSVが2本未満: {blend_files}"}

    merged = dfs[0]
    for d in dfs[1:]:
        merged = merged.merge(d, on="ID")
    target_cols = [c for c in merged.columns if c != "ID"]
    merged["target"] = merged[target_cols].mean(axis=1)

    pred_df = ctx.test[["ID"]].merge(merged[["ID", "target"]], on="ID", how="left")
    pred = pred_df["target"].values
    if np.isnan(pred).any():
        return {"ok": False, "message": "ID 不一致で NaN あり"}

    path = ctx.submissions_dir / "submission_improvement_04_blend.csv"
    result = _save_and_verify(ctx.test, pred, path)
    result["n_blended"] = len(dfs)
    return result


def run_02_similarity_te(
    ctx: ImprovementContext,
    k: int = 5,
) -> dict[str, Any]:
    """類似度TE（Jaccard top-k）。全 train で集合を構築。"""
    critic_movies: dict[Any, set] = defaultdict(set)
    movie_critics: dict[Any, set] = defaultdict(set)
    for _, row in ctx.train.iterrows():
        c, m = row["critic_name"], row["rotten_tomatoes_link"]
        critic_movies[c].add(m)
        movie_critics[m].add(c)
    critics = list(critic_movies.keys())
    movies = list(movie_critics.keys())

    def jaccard(s1: set, s2: set) -> float:
        u = len(s1 | s2) + 1e-9
        return (len(s1 & s2) + 1) / u

    topk_critic = {}
    for c in critics:
        sims = [(jaccard(critic_movies[c], critic_movies[c2]), c2) for c2 in critics if c2 != c]
        sims.sort(reverse=True, key=lambda x: x[0])
        topk_critic[c] = [x[1] for x in sims[:k]]

    topk_movie = {}
    for m in movies:
        sims = [(jaccard(movie_critics[m], movie_critics[m2]), m2) for m2 in movies if m2 != m]
        sims.sort(reverse=True, key=lambda x: x[0])
        topk_movie[m] = [x[1] for x in sims[:k]]

    movie_critic_mean: dict[Any, dict[Any, list]] = defaultdict(lambda: defaultdict(list))
    for _, row in ctx.train.iterrows():
        movie_critic_mean[row["rotten_tomatoes_link"]][row["critic_name"]].append(row["target"])
    global_mean = float(ctx.y.mean())

    def te_critic(row: pd.Series) -> float:
        c, m = row["critic_name"], row["rotten_tomatoes_link"]
        vals = []
        for c2 in topk_critic.get(c, [])[:k]:
            vals.extend(movie_critic_mean.get(m, {}).get(c2, []))
        return np.mean(vals) if vals else global_mean

    def te_movie(row: pd.Series) -> float:
        c, m = row["critic_name"], row["rotten_tomatoes_link"]
        vals = []
        for m2 in topk_movie.get(m, [])[:k]:
            vals.extend(movie_critic_mean.get(m2, {}).get(c, []))
        return np.mean(vals) if vals else global_mean

    train_02 = ctx.train.copy()
    test_02 = ctx.test.copy()
    train_02["te_sim_critic"] = train_02.apply(te_critic, axis=1)
    train_02["te_sim_movie"] = train_02.apply(te_movie, axis=1)
    test_02["te_sim_critic"] = test_02.apply(te_critic, axis=1)
    test_02["te_sim_movie"] = test_02.apply(te_movie, axis=1)

    feats = ctx.features + ["te_sim_critic", "te_sim_movie"]
    X_tr = train_02[feats]
    X_te = test_02[feats]

    model = lgb.LGBMClassifier(**ctx.lgb_params)
    model.fit(X_tr, ctx.y)
    pred = model.predict_proba(X_te)[:, 1]
    path = ctx.submissions_dir / "submission_improvement_02_similarity_te.csv"
    return _save_and_verify(ctx.test, pred, path)


def run_atmacup_ratio(
    ctx: ImprovementContext,
    base_feature: str,
    suffix: str,
) -> dict[str, Any]:
    """atmaCup 1位風: 値 / 批評家のその特徴の平均 を 1 列追加して学習・提出。

    1位の「集約前の値 / ユーザーごとの集約結果(平均)」に相当。分母は train のみで計算し、
    未知批評家は global 平均で割る。0 除算は 1.0 に置換。
    """
    if base_feature not in ctx.train.columns:
        return {"ok": False, "message": f"列がありません: {base_feature}"}
    tr = ctx.train.copy()
    te = ctx.test.copy()
    # 批評家ごとの平均（train のみで計算）
    critic_mean = tr.groupby("critic_name", observed=True)[base_feature].mean().to_dict()
    global_mean = tr[base_feature].astype(float).mean()
    tr["_cm"] = tr["critic_name"].map(critic_mean).fillna(global_mean)
    te["_cm"] = te["critic_name"].map(critic_mean).fillna(global_mean)
    col_ratio = f"ratio_{base_feature}_to_critic_mean"
    tr[col_ratio] = (tr[base_feature].astype(float) / tr["_cm"].replace(0, np.nan)).fillna(1.0).clip(0, 1e6)
    te[col_ratio] = (te[base_feature].astype(float) / te["_cm"].replace(0, np.nan)).fillna(1.0).clip(0, 1e6)
    feats = ctx.features + [col_ratio]
    X_tr = tr[feats].copy()
    X_te = te[feats].copy()
    for col in X_tr.columns:
        if not pd.api.types.is_numeric_dtype(X_tr[col]) and not pd.api.types.is_categorical_dtype(X_tr[col]):
            X_tr[col] = X_tr[col].astype("category")
            X_te[col] = X_te[col].astype("category")
    model = lgb.LGBMClassifier(**ctx.lgb_params)
    model.fit(X_tr, ctx.y)
    pred = model.predict_proba(X_te)[:, 1]
    path = ctx.submissions_dir / f"submission_atmacup_ratio_{suffix}.csv"
    return _save_and_verify(ctx.test, pred, path)


def _build_implicit_embeddings(
    ctx: ImprovementContext,
    method: str,
    factors: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """implicit で (critic, movie) 行列を分解し、ctx.train/test のコピーに埋め込み列を追加して返す。
    Returns:
        (train_with_emb, test_with_emb, feature_list)
    """
    try:
        import implicit
    except ImportError:
        raise ImportError("implicit 未導入: pip install implicit")

    train_df = ctx.train
    test_df = ctx.test
    critics = pd.concat([
        train_df["critic_name"].astype(str),
        test_df["critic_name"].astype(str),
    ]).unique()
    movies = pd.concat([
        train_df["rotten_tomatoes_link"].astype(str),
        test_df["rotten_tomatoes_link"].astype(str),
    ]).unique()
    c2i = {c: i for i, c in enumerate(critics)}
    m2j = {m: j for j, m in enumerate(movies)}
    n_c, n_m = len(c2i), len(m2j)

    rows, cols = [], []
    for _, r in train_df.iterrows():
        ci = c2i.get(str(r["critic_name"]))
        mj = m2j.get(str(r["rotten_tomatoes_link"]))
        if ci is not None and mj is not None:
            rows.append(ci)
            cols.append(mj)
    for _, r in test_df.iterrows():
        ci = c2i.get(str(r["critic_name"]))
        mj = m2j.get(str(r["rotten_tomatoes_link"]))
        if ci is not None and mj is not None:
            rows.append(ci)
            cols.append(mj)
    from scipy import sparse
    mat = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(n_c, n_m),
    )

    if method.lower() == "als":
        model_cf = implicit.als.AlternatingLeastSquares(factors=factors, random_state=42)
    elif method.lower() == "bpr":
        model_cf = implicit.bpr.BayesianPersonalizedRanking(factors=factors, random_state=42)
    else:
        raise ValueError(f"不明な method: {method}")
    model_cf.fit(mat)

    u_f = np.asarray(model_cf.user_factors, dtype=np.float32)
    i_f = np.asarray(model_cf.item_factors, dtype=np.float32)
    n_f = factors
    u_f = u_f[:, :n_f]
    i_f = i_f[:, :n_f]
    prefix_u = f"implicit_{method}_{factors}_u"
    prefix_i = f"implicit_{method}_{factors}_i"
    u_cols = [f"{prefix_u}_{k}" for k in range(n_f)]
    i_cols = [f"{prefix_i}_{k}" for k in range(n_f)]

    def get_emb(df: pd.DataFrame):
        cu = df["critic_name"].astype(str).map(lambda c: c2i.get(c, -1))
        mv = df["rotten_tomatoes_link"].astype(str).map(lambda m: m2j.get(m, -1))
        u_emb = np.zeros((len(df), n_f), dtype=np.float32)
        i_emb = np.zeros((len(df), n_f), dtype=np.float32)
        for idx in range(len(df)):
            ci, mj = cu.iloc[idx], mv.iloc[idx]
            if ci >= 0:
                u_emb[idx] = u_f[ci]
            if mj >= 0:
                i_emb[idx] = i_f[mj]
        return u_emb, i_emb

    u_tr, i_tr = get_emb(train_df)
    u_te, i_te = get_emb(test_df)
    tr = ctx.train.copy()
    te = ctx.test.copy()
    for k, c in enumerate(u_cols):
        tr[c] = u_tr[:, k]
        te[c] = u_te[:, k]
    for k, c in enumerate(i_cols):
        tr[c] = i_tr[:, k]
        te[c] = i_te[:, k]
    feats = ctx.features + u_cols + i_cols
    return tr, te, feats


def get_bpr_base(ctx: ImprovementContext, factors: int = 16) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """BPR 埋め込みを追加した train/test と特徴名リストを返す。2-hop 等の実験の土台用。"""
    return _build_implicit_embeddings(ctx, "bpr", factors)


def run_atmacup_implicit(
    ctx: ImprovementContext,
    method: str,
    factors: int,
    suffix: str,
) -> dict[str, Any]:
    """atmaCup 1位風: train+test の (critic, movie) 接続で implicit 埋め込みを学習し、特徴として追加。

    協調フィルタリングの考え方:
    - 批評家 = user、映画 = item。「誰がどの映画をレビューしたか」だけを 0/1 の交互行列にする（評価値は使わない = implicit feedback）。
    - ALS または BPR で行列を低ランク分解し、批評家ベクトル user_factors と映画ベクトル item_factors を得る。
    - 「似た批評家は似たベクトル」「似た映画は似たベクトル」になるので、その (批評家ベクトル | 映画ベクトル) を
      各行の追加特徴量として LGB に渡し、Fresh/Rotten を予測する。
    """
    try:
        tr, te, feats = _build_implicit_embeddings(ctx, method, factors)
    except ImportError as e:
        return {"ok": False, "message": str(e)}
    X_tr = tr[feats]
    X_te = te[feats]
    for col in X_tr.columns:
        if not pd.api.types.is_numeric_dtype(X_tr[col]) and not pd.api.types.is_categorical_dtype(X_tr[col]):
            X_tr[col] = X_tr[col].astype("category")
            X_te[col] = X_te[col].astype("category")
    model = lgb.LGBMClassifier(**ctx.lgb_params)
    model.fit(X_tr, ctx.y)
    pred = model.predict_proba(X_te)[:, 1]
    path = ctx.submissions_dir / f"submission_atmacup_implicit_{suffix}.csv"
    return _save_and_verify(ctx.test, pred, path)


def list_improvement_submissions(submissions_dir: Path) -> list[Path]:
    """改善候補で作成した提出ファイル一覧。"""
    return sorted(submissions_dir.glob("submission_improvement_*.csv"))
