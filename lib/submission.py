"""
提出用 CSV の保存・検証。全提出パイプラインで共通利用する。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def sanitize_predictions(pred: np.ndarray) -> np.ndarray:
    """
    予測値を提出可能な範囲に整える。
    NaN/Inf は 0.5 に、それ以外は [0, 1] に clip する。
    """
    out = np.asarray(pred, dtype=np.float64).ravel()
    out = np.clip(out, 0.0, 1.0)
    nan_inf = ~np.isfinite(out)
    if np.any(nan_inf):
        out[nan_inf] = 0.5
    return out


def save_submission(
    test_df: pd.DataFrame,
    pred: np.ndarray,
    path: str | Path,
    id_col: str = "ID",
    target_col: str = "target",
    sanitize: bool = True,
) -> None:
    """
    提出用 CSV を保存する。
    - test_df に id_col が無い場合は ValueError
    - pred の長さが test と一致しない場合は ValueError
    - sanitize=True のとき予測を [0,1] に clip し、NaN/Inf を 0.5 に置換してから保存
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if id_col not in test_df.columns:
        raise ValueError(
            f"test に ID 列がありません。列: {list(test_df.columns[:5])}..."
            if len(test_df.columns) > 5
            else f"test の列: {list(test_df.columns)}"
        )

    if len(pred) != len(test_df):
        raise ValueError(
            f"予測の長さ ({len(pred)}) が test の行数 ({len(test_df)}) と一致しません。"
        )

    if sanitize:
        pred = sanitize_predictions(pred)

    sub = pd.DataFrame({id_col: test_df[id_col].values, target_col: pred})
    sub.to_csv(path, index=False)


def verify_submission(
    path: str | Path,
    test_df: pd.DataFrame,
    id_col: str = "ID",
    target_col: str = "target",
) -> dict[str, Any]:
    """
    保存済み提出ファイルを検証する。
    戻り値: { "ok": bool, "rows": int, "rows_ok": bool, "cols_ok": bool,
              "target_ok": bool, "id_ok": bool, "message": str }
    """
    path = Path(path)
    result = {
        "ok": False,
        "path": str(path),
        "exists": path.exists(),
        "rows": 0,
        "rows_ok": False,
        "cols_ok": False,
        "target_ok": False,
        "id_ok": False,
        "message": "",
    }

    if not path.exists():
        result["message"] = "ファイルが存在しません。"
        return result

    if id_col not in test_df.columns:
        result["message"] = f"test に {id_col} 列がありません。"
        return result

    n_test = len(test_df)
    try:
        sub = pd.read_csv(path)
    except Exception as e:
        result["message"] = f"CSV 読み込みエラー: {e}"
        return result

    result["rows"] = len(sub)
    result["rows_ok"] = len(sub) == n_test
    result["cols_ok"] = list(sub.columns) == [id_col, target_col]
    result["target_ok"] = (
        target_col in sub.columns and sub[target_col].between(0, 1).all()
    )
    result["id_ok"] = (
        id_col in sub.columns
        and sub[id_col].equals(test_df[id_col].reset_index(drop=True))
    )
    result["ok"] = (
        result["rows_ok"]
        and result["cols_ok"]
        and result["target_ok"]
        and result["id_ok"]
    )
    result["message"] = (
        "OK"
        if result["ok"]
        else " / ".join(
            k for k in ["rows_ok", "cols_ok", "target_ok", "id_ok"] if not result[k]
        )
    )
    return result


def blend_two_submissions(
    path_a: str | Path,
    path_b: str | Path,
    out_path: str | Path,
    weight_a: float = 0.5,
    id_col: str = "ID",
    target_col: str = "target",
    test_ids: np.ndarray | pd.Series | None = None,
) -> dict[str, Any]:
    """
    2 本の提出 CSV を weight_a : (1 - weight_a) で加重平均し out_path に保存する。
    test_ids を渡すと行順をその ID 順に揃える（欠損は 0.5 で埋める）。
    """
    path_a, path_b, out_path = Path(path_a), Path(path_b), Path(out_path)
    if not path_a.exists():
        return {"ok": False, "path": str(out_path), "message": f"ファイルがありません: {path_a.name}"}
    if not path_b.exists():
        return {"ok": False, "path": str(out_path), "message": f"ファイルがありません: {path_b.name}"}
    weight_b = 1.0 - weight_a
    a = pd.read_csv(path_a)[[id_col, target_col]].rename(columns={target_col: "a"})
    b = pd.read_csv(path_b)[[id_col, target_col]].rename(columns={target_col: "b"})
    m = a.merge(b, on=id_col)
    m[target_col] = (weight_a * m["a"] + weight_b * m["b"]).astype(np.float64)
    if test_ids is not None:
        m = m.set_index(id_col).reindex(pd.Series(test_ids).values).reset_index()
        m[target_col] = m[target_col].fillna(0.5)
    m = m[[id_col, target_col]]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m[target_col] = sanitize_predictions(m[target_col].values)
    m.to_csv(out_path, index=False)
    return {"ok": True, "path": out_path, "message": "OK"}
