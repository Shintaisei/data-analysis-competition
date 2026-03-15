"""
train_high_potential_experiments.ipynb と同じ流れを .py で一括実行。
ノートを触らずに提出ファイルをすべて作成する用。

使い方（プロジェクトルートで）:
  python scripts/run_high_potential_experiments.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

from lib.improvement_candidates import (
    get_setup,
    get_bpr_base,
    run_atmacup_implicit,
    run_03_stacking,
    run_similar_movies_reviewed,
)
from lib.two_hop import run_experiment, TWO_HOP_REVIEW_COUNT
from lib.submission import save_submission, verify_submission


def blend_two(ctx, path_a: Path, path_b: Path, out_name: str) -> dict:
    if not path_a.exists() or not path_b.exists():
        return {"ok": False, "message": f"ファイル不足: {path_a.name} / {path_b.name}"}
    a = pd.read_csv(path_a)[["ID", "target"]].rename(columns={"target": "a"})
    b = pd.read_csv(path_b)[["ID", "target"]].rename(columns={"target": "b"})
    m = a.merge(b, on="ID")
    m["target"] = (m["a"] + m["b"]) / 2
    m = m.set_index("ID").reindex(ctx.test["ID"]).reset_index()
    out = ctx.submissions_dir / out_name
    save_submission(ctx.test, m["target"].values, out, sanitize=True)
    return verify_submission(out, ctx.test)


def main(output_dir: str = "outputs") -> None:
    ctx = get_setup(seed=42, output_dir=output_dir, use_best_pipeline=True)
    print(f"Train: {len(ctx.train):,}, Test: {len(ctx.test):,}, Features: {len(ctx.features)}")
    print(f"提出先: {ctx.submissions_dir}\n")

    path_bpr64 = ctx.submissions_dir / "submission_2hop_bpr64_only.csv"
    if not path_bpr64.exists():
        r = run_experiment(ctx, "bpr64_only", bpr_factors=64)
        print(f"  [bpr64_only] → {path_bpr64.name}  ({'OK' if r.get('ok') else r.get('message')})")
    else:
        print(f"  [bpr64_only] 既存ファイルを使用: {path_bpr64.name}")

    r_als = run_atmacup_implicit(ctx, "als", 64, "als64")
    print(f"  [ALS 64] → submission_atmacup_implicit_als64.csv  ({'OK' if r_als.get('ok') else r_als.get('message')})")

    r_count = run_experiment(ctx, "bpr64_count1", bpr_factors=64, use_2hop_cols=[TWO_HOP_REVIEW_COUNT])
    print(f"  [BPR64+count1] → submission_2hop_bpr64_count1.csv  ({'OK' if r_count.get('ok') else r_count.get('message')})")

    r_ratio = run_experiment(
        ctx, "bpr64_ratio_count1", bpr_factors=64, use_2hop_cols=[TWO_HOP_REVIEW_COUNT], use_2hop_ratio=True
    )
    print(f"  [BPR64+ratio+count1] → submission_2hop_bpr64_ratio_count1.csv  ({'OK' if r_ratio.get('ok') else r_ratio.get('message')})")

    r_bpr128 = run_experiment(ctx, "bpr128_only", bpr_factors=128)
    print(f"  [BPR 128] → submission_2hop_bpr128_only.csv  ({'OK' if r_bpr128.get('ok') else r_bpr128.get('message')})")

    r_blend_als = blend_two(ctx, path_bpr64, ctx.submissions_dir / "submission_atmacup_implicit_als64.csv", "submission_blend_bpr64_als64.csv")
    print(f"  [Blend BPR64+ALS64] → submission_blend_bpr64_als64.csv  ({'OK' if r_blend_als.get('ok') else r_blend_als.get('message')})")

    r_blend_bpr = blend_two(ctx, path_bpr64, ctx.submissions_dir / "submission_2hop_bpr128_only.csv", "submission_blend_bpr64_bpr128.csv")
    print(f"  [Blend BPR64+BPR128] → submission_blend_bpr64_bpr128.csv  ({'OK' if r_blend_bpr.get('ok') else r_blend_bpr.get('message')})")

    # BPR64 スタッキング（XGBoost は enable_categorical=True で category 列対応）
    train_df, test_df, feats = get_bpr_base(ctx, factors=64)
    for col in feats:
        if not pd.api.types.is_numeric_dtype(train_df[col]) and not pd.api.types.is_categorical_dtype(train_df[col]):
            train_df[col] = train_df[col].astype("category")
            test_df[col] = test_df[col].astype("category")
    ctx_bpr = ctx._replace(X=train_df[feats], X_test=test_df[feats], features=feats)
    r_stack = run_03_stacking(ctx_bpr, seed=None)
    if r_stack.get("path"):
        print(f"  [BPR64 Stacking] → {r_stack['path'].name}  ({'OK' if r_stack.get('ok') else r_stack.get('message')})")
    else:
        print(f"  [BPR64 Stacking] スキップ: {r_stack.get('message')}")

    try:
        r_sim = run_similar_movies_reviewed(ctx, top_k=20)
        if r_sim.get("path"):
            print(f"  [類似映画を何本レビュー] → {r_sim['path'].name}  ({'OK' if r_sim.get('ok') else r_sim.get('message')})")
        else:
            print(f"  [類似映画を何本レビュー] スキップ: {r_sim.get('message')}")
    except FileNotFoundError as e:
        print(f"  [類似映画を何本レビュー] スキップ: {e}")

    # 提出ファイル一覧・検証
    expected = [
        "submission_2hop_bpr64_only.csv",
        "submission_atmacup_implicit_als64.csv",
        "submission_2hop_bpr64_count1.csv",
        "submission_2hop_bpr64_ratio_count1.csv",
        "submission_2hop_bpr128_only.csv",
        "submission_blend_bpr64_als64.csv",
        "submission_blend_bpr64_bpr128.csv",
        "submission_improvement_03_stacking.csv",
        "submission_similar_movies_reviewed.csv",
    ]
    print("\n--- 提出ファイル一覧 ---")
    for name in expected:
        p = ctx.submissions_dir / name
        if not p.exists():
            print(f"  ✗ {name} （未作成）")
            continue
        ver = verify_submission(p, ctx.test)
        if ver["ok"]:
            print(f"  ✓ {name} （rows={ver['rows']:,}, ID・target 検証OK）")
        else:
            print(f"  ✗ {name} （{ver['message']}）")


if __name__ == "__main__":
    main()
