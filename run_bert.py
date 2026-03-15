#!/usr/bin/env python3
"""ModernBERT でテキスト分類し、提出 CSV を保存する。オプションで現最高精度とブレンドした提出も作成。

- デフォルト: ModernBERT（answerdotai/ModernBERT-base）。atmaCup 上位でよく使われる。
- 1位共有: microsoft/deberta-v3-large で LB 0.76242。
- 現最高: submission_blend_bpr64_count1_bpr128.csv（0.76591）とブレンドした提出を出力可能。
"""
import argparse
import warnings
warnings.filterwarnings("ignore")

from lib.improvement_candidates import get_setup, run_bert_deberta_submission, run_bert_blend_with_best


def main():
    p = argparse.ArgumentParser(description="ModernBERT で学習・提出し、必要なら最高精度とブレンド")
    p.add_argument("--model", default="answerdotai/ModernBERT-base",
                   help="モデル (ModernBERT / microsoft/deberta-v3-large 等)")
    p.add_argument("--no-blend", action="store_true", help="ブレンド提出を作らない（BERT 単体のみ）")
    p.add_argument("--weight-best", type=float, default=0.5,
                   help="ブレンド時の最高精度側の重み (0.5=50%% 最高 + 50%% BERT)")
    p.add_argument("--best", default="submission_blend_bpr64_count1_bpr128.csv",
                   help="最高精度提出ファイル名")
    args = p.parse_args()

    ctx = get_setup(seed=42, output_dir="outputs", use_best_pipeline=False)

    # 1) ModernBERT/DeBERTa で学習 → submission_modernbert.csv（または --model に合わせた名前）
    out_bert = "submission_modernbert.csv" if "ModernBERT" in args.model else "submission_bert_deberta.csv"
    r = run_bert_deberta_submission(
        ctx,
        model_name=args.model,
        n_folds=2,
        epochs=2,
        out_name=out_bert,
    )
    print("BERT:", "OK" if r.get("ok") else r.get("message"))

    # 2) 最高精度とブレンドした提出（submission_blend_best_bert.csv）
    if not args.no_blend:
        r2 = run_bert_blend_with_best(
            ctx,
            best_name=args.best,
            bert_name=out_bert,
            out_name="submission_blend_best_bert.csv",
            weight_best=args.weight_best,
        )
        print("ブレンド:", "OK" if r2.get("ok") else r2.get("message"))


if __name__ == "__main__":
    main()
