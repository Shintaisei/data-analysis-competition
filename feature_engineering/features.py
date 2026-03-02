"""
特徴量エンジニアリング: ホストベースラインと同一の create_features と FEATURES。
"""
import pandas as pd
import numpy as np


# ホストベースラインと同じ特徴リスト + original_release_date の分解（review_date と同じ扱い）
FEATURES = [
    "runtime",
    "top_critic",
    "review_year",
    "review_month",
    "review_dayofweek",
    "movie_age_days",
    "release_year",
    "release_month",
    "release_dayofweek",
    "content_rating",
    "publisher_name",
    "genre_Drama",
    "genre_Comedy",
    "genre_Action",
    "genre_Mystery",
    "genre_Fantasy",
    "genre_Romance",
    "genre_Horror",
    "genre_Documentary",
]


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    ホストベースライン (dojo4-host-baseline-v1) と完全に同一の特徴量作成。
    - 日付: pd.to_datetime のうえで year / month / dayofweek / movie_age_days
    - ジャンル: fillna("").str.contains で One-hot
    - カテゴリ: content_rating, publisher_name を fillna("missing").astype("category")
    """
    df = df.copy()

    # === 日付特徴量（ホストと同じ順序・同じ処理） ===
    df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce").astype("datetime64[ns]")
    df["review_year"] = df["review_date"].dt.year
    df["review_month"] = df["review_date"].dt.month
    df["review_dayofweek"] = df["review_date"].dt.dayofweek

    df["original_release_date"] = pd.to_datetime(df["original_release_date"], errors="coerce").astype("datetime64[ns]")
    # 公開日も review_date と同じく年・月・曜日に分解
    df["release_year"] = df["original_release_date"].dt.year
    df["release_month"] = df["original_release_date"].dt.month
    df["release_dayofweek"] = df["original_release_date"].dt.dayofweek
    # 差分は datetime 同士の減算（結果は Timedelta）。.dt.days で日数を取得
    days_diff = df["review_date"] - df["original_release_date"]
    df["movie_age_days"] = days_diff.dt.days
    df.loc[df["movie_age_days"] < 0, "movie_age_days"] = np.nan

    # === ジャンル特徴量 ===
    major_genres = [
        "Drama", "Comedy", "Action", "Mystery",
        "Fantasy", "Romance", "Horror", "Documentary",
    ]
    for genre in major_genres:
        df[f"genre_{genre}"] = df["genres"].fillna("").str.contains(genre, case=False).astype(int)

    # === カテゴリ特徴量 ===
    cat_cols = ["content_rating", "publisher_name"]
    for col in cat_cols:
        df[col] = df[col].fillna("missing").astype("category")

    return df
