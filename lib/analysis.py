"""
予測の当たり・外れを分析するユーティリティ。
「どこを当てているか／外しているか」をセグメント別に集計し、改善のヒントを得るために使う。
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def add_prediction_analysis(
    df: pd.DataFrame,
    y_true,
    y_pred,
    threshold: float = 0.5,
    pred_col: str = "pred",
    outcome_col: str = "outcome",
) -> pd.DataFrame:
    """
    データフレームに予測・正解ラベル・当たり外れ（TP/TN/FP/FN）を追加する。

    Parameters
    ----------
    df : pd.DataFrame
        元データ（index が y_true, y_pred と対応していること）
    y_true : array-like
        正解ラベル (0/1)
    y_pred : array-like
        予測確率（またはスコア）
    threshold : float
        予測を 0/1 に切る閾値
    pred_col : str
        予測値を入れる列名
    outcome_col : str
        TP/TN/FP/FN を入れる列名

    Returns
    -------
    pd.DataFrame
        pred, pred_label, correct, outcome を追加したコピー
    """
    df = df.copy()
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    pred_label = (y_pred >= threshold).astype(int)

    df[pred_col] = y_pred
    df["pred_label"] = pred_label
    df["correct"] = (df["pred_label"] == y_true).astype(int)
    # outcome: TP, TN, FP, FN
    outcome = np.where(
        y_true == 1,
        np.where(pred_label == 1, "TP", "FN"),
        np.where(pred_label == 1, "FP", "TN"),
    )
    df[outcome_col] = outcome
    return df


def summarize_errors_by(
    df: pd.DataFrame,
    group_col: str,
    y_true_col: str = "target",
    pred_col: str = "pred",
    outcome_col: str = "outcome",
    min_count: int = 50,
) -> pd.DataFrame:
    """
    セグメント（group_col でグループ化）ごとに予測の良し悪しを集計する。

    Returns
    -------
    pd.DataFrame
        列: group 値, n, n_TP, n_TN, n_FP, n_FN, accuracy, auc(可能なら), FP_rate, FN_rate, ...
    """
    rows = []
    for name, g in df.groupby(group_col, dropna=False):
        n = len(g)
        if n < min_count:
            continue
        y_true = g[y_true_col].values
        y_pred = g[pred_col].values
        out = g[outcome_col]

        n_tp = (out == "TP").sum()
        n_tn = (out == "TN").sum()
        n_fp = (out == "FP").sum()
        n_fn = (out == "FN").sum()

        acc = (n_tp + n_tn) / n if n else 0
        fp_rate = n_fp / (n_tn + n_fp) if (n_tn + n_fp) > 0 else np.nan
        fn_rate = n_fn / (n_tp + n_fn) if (n_tp + n_fn) > 0 else np.nan

        try:
            auc = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) >= 2 else np.nan
        except Exception:
            auc = np.nan

        pos_rate = y_true.mean()
        pred_pos_rate = (y_pred >= 0.5).mean()

        rows.append({
            group_col: name,
            "n": n,
            "n_TP": n_tp,
            "n_TN": n_tn,
            "n_FP": n_fp,
            "n_FN": n_fn,
            "accuracy": acc,
            "auc": auc,
            "FP_rate": fp_rate,
            "FN_rate": fn_rate,
            "pos_rate": pos_rate,
            "pred_pos_rate": pred_pos_rate,
        })
    return pd.DataFrame(rows)


def run_full_analysis(
    df: pd.DataFrame,
    y_true,
    y_pred,
    group_cols=None,
    threshold: float = 0.5,
    min_count: int = 50,
):
    """
    予測分析を一括実行する。
    - add_prediction_analysis で df に当たり外れを追加
    - 各 group_col で summarize_errors_by を実行してリストで返す

    Returns
    -------
    analysis_df : pd.DataFrame
        当たり外れ列を追加したデータフレーム
    summaries : dict[str, pd.DataFrame]
        group_col をキーにしたセグメント別集計の辞書
    """
    if group_cols is None:
        group_cols = ["review_year", "content_rating"]
        # データに存在する列だけ使う
        group_cols = [c for c in group_cols if c in df.columns]

    analysis_df = add_prediction_analysis(df, y_true, y_pred, threshold=threshold)
    summaries = {}
    for col in group_cols:
        if col not in df.columns:
            continue
        summaries[col] = summarize_errors_by(
            analysis_df, col, pred_col="pred", outcome_col="outcome", min_count=min_count
        )
    return analysis_df, summaries
