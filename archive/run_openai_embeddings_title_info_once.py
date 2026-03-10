#!/usr/bin/env python3
"""
OpenAI API で「タイトル + 映画情報」を映画単位で1回だけベクトル化し、outputs/embeddings/ に保存する。

使い方:
  1. config/openai_api_key.txt の1行目に OpenAI の API キー（sk- で始まる）を設定する。
  2. コスト確認: python archive/run_openai_embeddings_title_info_once.py --estimate
  3. 実行: python archive/run_openai_embeddings_title_info_once.py  （プロジェクトルートで）

保存先: outputs/embeddings/movie_title_info_embeddings.parquet（または .pkl）
既存の movie_info のみの embedding とは別ファイルで保存する。
"""
import argparse
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from lib.openai_embeddings import (
    DEFAULT_TITLE_INFO_EMBEDDINGS_PATH,
    compute_and_save_title_info_embeddings,
    estimate_title_info_embedding_cost,
)


def _print_cost_estimate(info: dict) -> None:
    print("--- コスト見積もり（タイトル+映画情報、API はまだ呼びません）---")
    print(f"  対象映画数:     {info['n_movies']:,} 本")
    print(f"  推定トークン数: {info['estimated_tokens']:,}")
    print(f"  API 呼び出し:   {info['n_api_calls']} 回")
    print(f"  料金目安:       ${info['estimated_usd']:.6f} USD（約 {info['estimated_usd'] * 150:.2f} 円、1USD=150円の場合）")
    print("---")


def main():
    parser = argparse.ArgumentParser(
        description="1回だけ OpenAI で「タイトル+映画情報」をベクトル化して保存"
    )
    parser.add_argument("--force", action="store_true", help="既存ファイルがあっても再計算する")
    parser.add_argument("--estimate", action="store_true", help="API は呼ばず、コスト見積もりのみ表示")
    args = parser.parse_args()

    if args.estimate:
        info = estimate_title_info_embedding_cost()
        _print_cost_estimate(info)
        print("実際に実行する場合は --estimate を付けずに実行してください。")
        return

    save_path = DEFAULT_TITLE_INFO_EMBEDDINGS_PATH
    pkl_path = save_path.with_suffix(".pkl")
    if (save_path.exists() or pkl_path.exists()) and not args.force:
        p = save_path if save_path.exists() else pkl_path
        print(f"既に存在するためスキップ: {p}")
        print("再計算する場合は --force を付けて実行してください。")
        return

    info = estimate_title_info_embedding_cost()
    _print_cost_estimate(info)
    print("OpenAI API で「タイトル + 映画情報」を映画単位でベクトル化しています...")
    df = compute_and_save_title_info_embeddings(force=args.force)
    out = save_path if save_path.exists() else pkl_path
    print(f"保存しました: {out} (行数={len(df)}, 次元={len(df.columns)-1})")
    print("学習では load_movie_title_info_embeddings() または path= でこのファイルを指定してください。")


if __name__ == "__main__":
    main()
