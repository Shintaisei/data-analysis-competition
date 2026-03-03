"""
ベースライン用パイプライン: データ読み込み〜特徴量構築までを関数化。
train_baseline.ipynb の処理と同一。改変のベースとして使う。
"""
import numpy as np
import pandas as pd

from preprocess import load_train_test
from feature_engineering import create_features
from .encodings import movie_info_meta

# ベースラインで使う 38 特徴量（3C3 ＋ テキストメタ）
BASELINE_FEATURES = [
    "rotten_tomatoes_link",
    "critic_name",
    "top_critic",
    "publisher_name",
    "movie_title",
    "movie_info",
    "content_rating",
    "directors",
    "authors",
    "actors",
    "runtime",
    "production_company",
    "review_year",
    "review_month",
    "review_dayofweek",
    "release_year",
    "release_month",
    "release_dayofweek",
    "movie_age_days",
    "genre_Drama",
    "genre_Comedy",
    "genre_Action",
    "genre_Mystery",
    "genre_Fantasy",
    "genre_Romance",
    "genre_Horror",
    "genre_Documentary",
    "critic_name_te_ts",
    "production_company_te_ts",
    "critic_name_te_ts_bin",
    "production_company_te_ts_bin",
    "runtime_bin",
    "movie_age_bin",
    "release_decade",
    "movie_info_len",
    "movie_info_word_count",
    "movie_title_len",
    "movie_title_word_count",
]


def _ts_te_col(df_tr, df_te, col, target_name="target", m=10):
    """時系列TE: tr は「その行より前」のみで平均、te は tr のカテゴリ別スムージング平均でマッピング。"""
    global_mean = float(df_tr[target_name].mean())
    tr_s = df_tr.sort_values("review_date")
    g = tr_s.groupby(col)[target_name]
    past_sum = g.cumsum() - tr_s[target_name]
    past_count = g.cumcount()
    te_tr = np.where(past_count > 0, (past_sum + m * global_mean) / (past_count + m), global_mean)
    ser_tr = pd.Series(te_tr, index=tr_s.index)
    agg = df_tr.groupby(col)[target_name].agg(["mean", "count"])
    agg["smooth"] = (agg["count"] * agg["mean"] + m * global_mean) / (agg["count"] + m)
    map_ = agg["smooth"].to_dict()
    te_arr = (
        df_te[col].astype(str).map(map_).fillna(global_mean).values
        if df_te is not None and len(df_te)
        else np.array([])
    )
    return ser_tr, te_arr


def _movie_age_bin(x):
    if pd.isna(x) or x < 0:
        return 0
    if x < 365:
        return 1
    if x < 365 * 5:
        return 2
    if x < 365 * 20:
        return 3
    return 4


def add_3c3_and_text_meta(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """
    train / test に 3C3（時系列TE・ビン）とテキストメタを in-place で追加する。
    create_features と category/runtime の前処理を済ませた DataFrame を渡すこと。
    """
    for col in ["critic_name", "production_company"]:
        if col not in train.columns:
            continue
        st, ta = _ts_te_col(train, test, col, "target", m=10)
        train[f"{col}_te_ts"] = st.reindex(train.index).fillna(train["target"].mean())
        test[f"{col}_te_ts"] = ta

    for c in ["critic_name_te_ts", "production_company_te_ts"]:
        if c not in train.columns:
            continue
        train[c + "_bin"] = (
            pd.cut(train[c], bins=[0, 0.33, 0.67, 1.01], labels=[0, 1, 2]).astype(float).fillna(1)
        )
        test[c + "_bin"] = (
            pd.cut(test[c], bins=[0, 0.33, 0.67, 1.01], labels=[0, 1, 2]).astype(float).fillna(1)
        )

    train["runtime_bin"] = (
        pd.cut(train["runtime"], bins=[0, 90, 120, 150, 1000], labels=[0, 1, 2, 3])
        .astype(float)
        .fillna(1)
    )
    test["runtime_bin"] = (
        pd.cut(test["runtime"], bins=[0, 90, 120, 150, 1000], labels=[0, 1, 2, 3])
        .astype(float)
        .fillna(1)
    )
    train["movie_age_bin"] = train["movie_age_days"].apply(_movie_age_bin)
    test["movie_age_bin"] = test["movie_age_days"].apply(_movie_age_bin)
    train["release_decade"] = (train["release_year"] // 10 * 10).fillna(1990)
    test["release_decade"] = (test["release_year"] // 10 * 10).fillna(1990)

    movie_info_meta(train, pd.DataFrame(), test)
    train["movie_title_len"] = train["movie_title"].astype(str).str.len()
    test["movie_title_len"] = test["movie_title"].astype(str).str.len()
    train["movie_title_word_count"] = (
        train["movie_title"].astype(str).str.split().str.len().fillna(0).astype(int)
    )
    test["movie_title_word_count"] = (
        test["movie_title"].astype(str).str.split().str.len().fillna(0).astype(int)
    )


def prepare_baseline_data(train: pd.DataFrame, test: pd.DataFrame):
    """
    create_features 済みの train / test を受け、カテゴリ・欠損処理と 3C3・テキストメタを追加する。
    戻り値: (train, test) のタプル（in-place で更新した同じオブジェクト）。
    """
    cat_cols = [
        "rotten_tomatoes_link",
        "critic_name",
        "movie_title",
        "movie_info",
        "directors",
        "authors",
        "actors",
        "production_company",
    ]
    for col in cat_cols:
        if col in train.columns and col in test.columns:
            train[col] = train[col].fillna("missing").astype("category")
            test[col] = test[col].fillna("missing").astype("category")

    if "runtime" in train.columns and train["runtime"].isna().any():
        med = train["runtime"].median()
        train["runtime"] = train["runtime"].fillna(med)
        test["runtime"] = test["runtime"].fillna(med)

    add_3c3_and_text_meta(train, test)
    return train, test


def get_baseline_data():
    """load_train_test → create_features → prepare_baseline_data まで一括で実行し (train, test) を返す。"""
    train, test = load_train_test()
    train = create_features(train)
    test = create_features(test)
    train, test = prepare_baseline_data(train, test)
    return train, test
