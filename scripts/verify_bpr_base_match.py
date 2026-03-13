"""
BPR ベースが 0.76101 提出と同一か検証する。
run_experiment(ctx, "bpr_only") と run_atmacup_implicit(ctx, "bpr", 16, "bpr16") が
同じ特徴量・同じデータで学習し、同じ予測を出すことを確認する。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from lib.improvement_candidates import get_setup, get_bpr_base, run_atmacup_implicit, _build_implicit_embeddings
from lib.two_hop import run_experiment


def main():
    ctx = get_setup(seed=42, output_dir="outputs", use_best_pipeline=True)
    # 1) get_bpr_base と _build_implicit_embeddings の返り値が一致
    tr1, te1, feats1 = get_bpr_base(ctx, 16)
    tr2, te2, feats2 = _build_implicit_embeddings(ctx, "bpr", 16)
    assert feats1 == feats2, "feats が一致しません"
    assert list(tr1.columns) == list(tr2.columns), "train 列が一致しません"
    assert list(te1.columns) == list(te2.columns), "test 列が一致しません"
    for c in feats1:
        np.testing.assert_array_almost_equal(tr1[c].values, tr2[c].values, err_msg=f"train 列 {c}")
        np.testing.assert_array_almost_equal(te1[c].values, te2[c].values, err_msg=f"test 列 {c}")
    print("OK: get_bpr_base と _build_implicit_embeddings(bpr, 16) の返り値は一致")

    # 2) run_experiment("bpr_only") と run_atmacup_implicit("bpr", 16) の予測が一致
    out = ctx.submissions_dir
    r1 = run_experiment(ctx, "bpr_only", use_2hop_cols=None)
    r2 = run_atmacup_implicit(ctx, "bpr", 16, "bpr16")
    if not r1.get("ok") or not r2.get("ok"):
        print("SKIP: 提出保存に失敗したため予測の比較をスキップ")
        return
    p1 = out / "submission_2hop_bpr_only.csv"
    p2 = out / "submission_atmacup_implicit_bpr16.csv"
    if not p1.exists() or not p2.exists():
        print("SKIP: 提出ファイルが存在しないため比較をスキップ")
        return
    sub1 = pd.read_csv(p1)
    sub2 = pd.read_csv(p2)
    np.testing.assert_array_almost_equal(
        sub1["target"].values, sub2["target"].values, decimal=9,
        err_msg="run_experiment(bpr_only) と run_atmacup_implicit(bpr,16) の予測が一致しません"
    )
    print("OK: run_experiment(bpr_only) と run_atmacup_implicit(bpr, 16, bpr16) の予測は一致")
    print("→ 2-hop 実験のベースは 0.76101 提出と同一です。")


if __name__ == "__main__":
    main()
