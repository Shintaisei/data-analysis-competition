"""
実験用エンコーディング集。時系列CV用に「tr のみで統計を計算し val/te に適用」する関数。
各関数は tr_df, val_df, te_df を in-place で更新し、追加した列名のリストを返す。
"""
import numpy as np
import pandas as pd


def ts_te(df_tr, df_val, df_te, col, target_name="target", m=10):
    """
    時系列 Target Encoding: tr は「その行より前」のみで平均、val/te は tr 全体のカテゴリ別スムージング平均。
    追加列: {col}_te_ts
    """
    if col not in df_tr.columns:
        return []
    global_mean = float(df_tr[target_name].mean())
    tr_s = df_tr.sort_values("review_date")
    g = tr_s.groupby(col)[target_name]
    past_sum = g.cumsum() - tr_s[target_name]
    past_count = g.cumcount()
    te_tr = np.where(
        past_count > 0,
        (past_sum + m * global_mean) / (past_count + m),
        global_mean,
    )
    ser_tr = pd.Series(te_tr, index=tr_s.index)
    agg = df_tr.groupby(col)[target_name].agg(["mean", "count"])
    agg["smooth"] = (agg["count"] * agg["mean"] + m * global_mean) / (agg["count"] + m)
    map_ = agg["smooth"].to_dict()
    name = f"{col}_te_ts"
    df_tr[name] = ser_tr.reindex(df_tr.index).fillna(global_mean)
    if len(df_val):
        df_val[name] = df_val[col].astype(str).map(map_).fillna(global_mean).values
    if df_te is not None and len(df_te):
        df_te[name] = df_te[col].astype(str).map(map_).fillna(global_mean).values
    return [name]


def ts_te_binned(df_tr, df_val, df_te, col, target_name="target", m=10, bins=(0, 0.33, 0.67, 1.01)):
    """
    時系列 TE を追加し、さらにその値を 3 ビンにした列を追加。単一軸依存を弱める。
    追加列: {col}_te_ts, {col}_te_ts_bin
    """
    added = ts_te(df_tr, df_val, df_te, col, target_name, m)
    if not added:
        return []
    name_te = added[0]
    name_bin = f"{col}_te_ts_bin"
    labels = [0, 1, 2]
    df_tr[name_bin] = pd.cut(df_tr[name_te], bins=bins, labels=labels).astype(float).fillna(1)
    if len(df_val):
        df_val[name_bin] = pd.cut(df_val[name_te], bins=bins, labels=labels).astype(float).fillna(1)
    if df_te is not None and len(df_te):
        df_te[name_bin] = pd.cut(df_te[name_te], bins=bins, labels=labels).astype(float).fillna(1)
    return added + [name_bin]


def freq(df_tr, df_val, df_te, col):
    """出現回数（頻度）を追加。追加列: {col}_freq"""
    if col not in df_tr.columns:
        return []
    vc = df_tr[col].astype(str).value_counts()
    name = f"{col}_freq"
    df_tr[name] = df_tr[col].astype(str).map(vc).fillna(0).astype(int)
    if len(df_val):
        df_val[name] = df_val[col].astype(str).map(vc).fillna(0).astype(int).values
    if df_te is not None and len(df_te):
        df_te[name] = df_te[col].astype(str).map(vc).fillna(0).astype(int).values
    return [name]


def movie_info_meta(df_tr, df_val, df_te):
    """movie_info の長さ・語数を追加（movie_info は残す）。追加列: movie_info_len, movie_info_word_count"""
    if "movie_info" not in df_tr.columns:
        return []
    df_tr["movie_info_len"] = df_tr["movie_info"].astype(str).str.len()
    df_tr["movie_info_word_count"] = (
        df_tr["movie_info"].astype(str).str.split().str.len().fillna(0).astype(int)
    )
    if len(df_val):
        df_val["movie_info_len"] = df_val["movie_info"].astype(str).str.len()
        df_val["movie_info_word_count"] = (
            df_val["movie_info"].astype(str).str.split().str.len().fillna(0).astype(int)
        )
    if df_te is not None and len(df_te):
        df_te["movie_info_len"] = df_te["movie_info"].astype(str).str.len()
        df_te["movie_info_word_count"] = (
            df_te["movie_info"].astype(str).str.split().str.len().fillna(0).astype(int)
        )
    return ["movie_info_len", "movie_info_word_count"]


def per_movie_ts(df_tr, df_val, df_te, movie_col="rotten_tomatoes_link", target_name="target"):
    """
    映画（rotten_tomatoes_link）ごとに時系列で「その行より前のレビュー数」「その行より前の Fresh 率」を追加。
    追加列: movie_review_count_ts, movie_fresh_rate_ts
    """
    if movie_col not in df_tr.columns or target_name not in df_tr.columns:
        return []
    tr_s = df_tr.sort_values("review_date")
    g = tr_s.groupby(movie_col)
    past_count = g.cumcount()
    past_sum = g[target_name].cumsum() - tr_s[target_name]
    global_mean = float(df_tr[target_name].mean())
    df_tr["movie_review_count_ts"] = past_count.reindex(df_tr.index).fillna(0).astype(int)
    rate_ser = np.where(past_count > 0, past_sum / past_count, global_mean)
    df_tr["movie_fresh_rate_ts"] = pd.Series(rate_ser, index=tr_s.index).reindex(df_tr.index).fillna(global_mean)

    agg = df_tr.groupby(movie_col)[target_name].agg(["sum", "count"])
    rate_map = (agg["sum"] / agg["count"]).to_dict()
    count_map = agg["count"].to_dict()
    if len(df_val):
        df_val["movie_review_count_ts"] = df_val[movie_col].astype(str).map(count_map).fillna(0).astype(int).values
        df_val["movie_fresh_rate_ts"] = df_val[movie_col].astype(str).map(rate_map).fillna(global_mean).values
    if df_te is not None and len(df_te):
        df_te["movie_review_count_ts"] = df_te[movie_col].astype(str).map(count_map).fillna(0).astype(int).values
        df_te["movie_fresh_rate_ts"] = df_te[movie_col].astype(str).map(rate_map).fillna(global_mean).values
    return ["movie_review_count_ts", "movie_fresh_rate_ts"]


def missing_flags(df_tr, df_val, df_te):
    """欠損フラグ 2 値。追加列: is_runtime_missing, is_movie_age_days_missing"""
    extra = []
    if "runtime" in df_tr.columns:
        df_tr["is_runtime_missing"] = (df_tr["runtime"].isna()).astype(int)
        if len(df_val):
            df_val["is_runtime_missing"] = (df_val["runtime"].isna()).astype(int)
        if df_te is not None and len(df_te):
            df_te["is_runtime_missing"] = (df_te["runtime"].isna()).astype(int)
        extra.append("is_runtime_missing")
    if "movie_age_days" in df_tr.columns:
        df_tr["is_movie_age_days_missing"] = (df_tr["movie_age_days"].isna()).astype(int)
        if len(df_val):
            df_val["is_movie_age_days_missing"] = (df_val["movie_age_days"].isna()).astype(int)
        if df_te is not None and len(df_te):
            df_te["is_movie_age_days_missing"] = (df_te["movie_age_days"].isna()).astype(int)
        extra.append("is_movie_age_days_missing")
    return extra


def loo(df_tr, df_val, df_te, col, target_name="target", m=5):
    """
    Leave-One-Out: 自分を除いた同カテゴリの target 平均。val/te は tr のカテゴリ別スムージング平均。
    追加列: {col}_loo
    """
    if col not in df_tr.columns:
        return []
    global_mean = float(df_tr[target_name].mean())
    g = df_tr.groupby(col)[target_name]
    s_sum, s_cnt = g.transform("sum"), g.transform("count")
    y = df_tr[target_name].values
    loo_raw = np.where(s_cnt > 1, (s_sum - y) / (s_cnt - 1), global_mean)
    ser_tr = pd.Series((s_cnt * loo_raw + m * global_mean) / (s_cnt + m), index=df_tr.index)
    agg = df_tr.groupby(col)[target_name].agg(["mean", "count"])
    agg["smooth"] = (agg["count"] * agg["mean"] + m * global_mean) / (agg["count"] + m)
    map_ = agg["smooth"].to_dict()
    name = f"{col}_loo"
    df_tr[name] = ser_tr
    if len(df_val):
        df_val[name] = df_val[col].astype(str).map(map_).fillna(global_mean).values
    if df_te is not None and len(df_te):
        df_te[name] = df_te[col].astype(str).map(map_).fillna(global_mean).values
    return [name]


def add_freq_multi(df_tr, df_val, df_te, cols):
    """複数列に頻度を追加。追加列: {col}_freq for each col in cols"""
    out = []
    for col in cols:
        out.extend(freq(df_tr, df_val, df_te, col))
    return out


def ts_te_multi(df_tr, df_val, df_te, cols, target_name="target", m=10):
    """複数列に時系列 TE を追加。追加列: {col}_te_ts for each col in cols"""
    out = []
    for col in cols:
        out.extend(ts_te(df_tr, df_val, df_te, col, target_name, m))
    return out
