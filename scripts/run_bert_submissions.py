"""
train_light_experiments.ipynb の #11–20（BERT 系）だけを別ファイルで実行する。
ノートでやるとセル再実行やカーネル落ちでやり直しが発生するため、再開しやすいようにスクリプト化。

使い方（プロジェクトルートで）:
  python scripts/run_bert_submissions.py
  python scripts/run_bert_submissions.py --output-dir outputs
  python scripts/run_bert_submissions.py --no-skip-existing   # 既存提出があっても再実行

#11–14: BERT/ModernBERT 単体（fold キャッシュで再開可）
#15–18, #20: 最高精度提出 + BERT のブレンド
#19: 4本均等ブレンド（bpr64_only, als64, bpr128, ModernBERT）
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

# プロジェクトルートを path に追加（どこから実行しても lib を import できるように）
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pandas as pd

warnings.filterwarnings("ignore")

from lib.improvement_candidates import (
    get_setup,
    run_bert_deberta_submission,
    run_bert_blend_with_best,
)
from lib.submission import save_submission


def main(output_dir: str = "outputs", skip_existing: bool = True) -> None:
    out = Path(output_dir)
    submissions_dir = out / "submissions"
    cache_dir = out / "bert_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    ctx_bert = get_setup(seed=42, output_dir=output_dir, use_best_pipeline=False)
    ctx = get_setup(seed=42, output_dir=output_dir, use_best_pipeline=True)

    print(f"提出先: {submissions_dir}")
    print(f"BERT キャッシュ: {cache_dir}")
    print(f"既存スキップ: {skip_existing}\n")

    best_name = "submission_blend_bpr64_count1_bpr128.csv"

    # --- #11–14: BERT 単体 ---
    jobs_11_14 = [
        (11, "submission_modernbert.csv", {"out_name": "submission_modernbert.csv", "cache_name": "modernbert"}),
        (12, "submission_modernbert_time.csv", {"cv_strategy": "time", "out_name": "submission_modernbert_time.csv", "cache_name": "modernbert_time"}),
        (13, "submission_deberta_v3_base.csv", {"model_name": "microsoft/deberta-v3-base", "out_name": "submission_deberta_v3_base.csv", "cache_name": "deberta_v3_base"}),
        (14, "submission_modernbert_nofill.csv", {"use_fill_map": False, "out_name": "submission_modernbert_nofill.csv", "cache_name": "modernbert_nofill"}),
    ]
    for num, fname, kwargs in jobs_11_14:
        path = submissions_dir / fname
        if skip_existing and path.exists():
            print(f"  [{num}] 既存: {fname}")
            continue
        r = run_bert_deberta_submission(
            ctx_bert,
            cache_dir=cache_dir,
            **kwargs,
        )
        status = "OK" if r.get("ok") else r.get("message", "?")
        print(f"  [{num}] → {fname}  ({status})")

    # --- #15–18, #20: 最高精度 + BERT ブレンド ---
    blend_jobs = [
        (15, "submission_blend_best_bert.csv", "submission_modernbert.csv", 0.5),
        (16, "submission_blend_best_bert_07.csv", "submission_modernbert.csv", 0.7),
        (17, "submission_blend_best_bert_03.csv", "submission_modernbert.csv", 0.3),
        (18, "submission_blend_best_deberta.csv", "submission_deberta_v3_base.csv", 0.5),
        (20, "submission_blend_best_bert_06.csv", "submission_modernbert.csv", 0.6),
    ]
    for num, out_name, bert_name, weight_best in blend_jobs:
        path = submissions_dir / out_name
        if skip_existing and path.exists():
            print(f"  [{num}] 既存: {out_name}")
            continue
        r = run_bert_blend_with_best(
            ctx,
            best_name=best_name,
            bert_name=bert_name,
            out_name=out_name,
            weight_best=weight_best,
        )
        status = "OK" if r.get("ok") else r.get("message", "?")
        print(f"  [{num}] → {out_name}  ({status})")

    # --- #19: 4本均等ブレンド ---
    out_19 = submissions_dir / "submission_blend_weighted_4.csv"
    paths_4 = [
        submissions_dir / "submission_2hop_bpr64_only.csv",
        submissions_dir / "submission_atmacup_implicit_als64.csv",
        submissions_dir / "submission_2hop_bpr128_only.csv",
        submissions_dir / "submission_modernbert.csv",
    ]
    if skip_existing and out_19.exists():
        print(f"  [19] 既存: submission_blend_weighted_4.csv")
    elif all(p.exists() for p in paths_4):
        dfs = [pd.read_csv(p)[["ID", "target"]].rename(columns={"target": p.stem}) for p in paths_4]
        m = dfs[0]
        for d in dfs[1:]:
            m = m.merge(d, on="ID")
        m["target"] = m[[c for c in m.columns if c != "ID"]].mean(axis=1)
        m = m[["ID", "target"]].set_index("ID").reindex(ctx.test["ID"]).reset_index()
        save_submission(ctx.test, m["target"].values, out_19, sanitize=True)
        print(f"  [19] → submission_blend_weighted_4.csv")
    else:
        missing = [p.name for p in paths_4 if not p.exists()]
        print(f"  [19] スキップ: ファイル不足 {missing}")

    print("\n完了.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BERT 系提出 #11–20 を実行（既存はスキップ）")
    parser.add_argument("--output-dir", default="outputs", help="出力ディレクトリ (default: outputs)")
    parser.add_argument("--no-skip-existing", action="store_true", help="既存提出があっても再実行する")
    args = parser.parse_args()
    main(output_dir=args.output_dir, skip_existing=not args.no_skip_existing)
