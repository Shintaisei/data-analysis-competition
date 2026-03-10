#!/usr/bin/env python3
"""
OpenAI API で movie_info を映画単位で1回だけベクトル化し、outputs/embeddings/ に保存する。

使い方:
  1. config/openai_api_key.txt を開き、1行目に OpenAI の API キー（sk- で始まる）を貼り付けて保存する。
     初回は config/openai_api_key.example.txt をコピーして openai_api_key.txt にリネームしてもよい。
  2. コストを確認したい場合: python archive/run_openai_embeddings_once.py --estimate  （API は呼ばない）
  3. 実行: python archive/run_openai_embeddings_once.py  （プロジェクトルートで）

既に outputs/embeddings/movie_info_embeddings.parquet がある場合はスキップする。
環境変数 OPENAI_API_KEY が設定されていれば、そちらを優先して使う。
方針・実行方法: docs/PROJECT_MIND.md
"""
import argparse
import sys
from pathlib import Path

# archive/ から実行してもプロジェクトルートを参照できるようにする
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from lib.openai_embeddings import (
    DEFAULT_EMBEDDINGS_PATH,
    compute_and_save_movie_info_embeddings,
    estimate_embedding_cost,
)


def _print_cost_estimate(info: dict) -> None:
    """コスト見積もりを表示する。"""
    print("--- コスト見積もり（API はまだ呼びません）---")
    print(f"  対象映画数:     {info['n_movies']:,} 本")
    print(f"  推定トークン数: {info['estimated_tokens']:,}")
    print(f"  API 呼び出し:   {info['n_api_calls']} 回（{info['n_api_calls']} バッチ）")
    print(f"  料金目安:       ${info['estimated_usd']:.6f} USD（約 {info['estimated_usd'] * 150:.2f} 円、1USD=150円の場合）")
    print(f"  単価:           ${info['price_per_1k_tokens_usd']:.6f} / 1,000 tokens（text-embedding-3-small）")
    print("---")


def main():
    parser = argparse.ArgumentParser(description="1回だけ OpenAI で movie_info をベクトル化して保存")
    parser.add_argument(
        "--force",
        action="store_true",
        help="既存ファイルがあっても再計算する",
    )
    parser.add_argument(
        "--estimate",
        action="store_true",
        help="API は呼ばず、かかるコストの見積もりだけ表示して終了する",
    )
    args = parser.parse_args()

    if args.estimate:
        info = estimate_embedding_cost()
        _print_cost_estimate(info)
        print("実際に実行する場合は --estimate を付けずに run_openai_embeddings_once.py を実行してください。")
        return

    cache_path = DEFAULT_EMBEDDINGS_PATH
    cache_pkl = cache_path.with_suffix(".pkl")
    if (cache_path.exists() or cache_pkl.exists()) and not args.force:
        p = cache_path if cache_path.exists() else cache_pkl
        print(f"既に存在するためスキップ: {p}")
        print("再計算する場合は --force を付けて実行してください。")
        return

    info = estimate_embedding_cost()
    _print_cost_estimate(info)
    print("OpenAI API で movie_info を映画単位でベクトル化しています...")
    df = compute_and_save_movie_info_embeddings(force=args.force)
    saved = DEFAULT_EMBEDDINGS_PATH if DEFAULT_EMBEDDINGS_PATH.exists() else DEFAULT_EMBEDDINGS_PATH.with_suffix(".pkl")
    print(f"保存しました: {saved} (行数={len(df)}, 次元={len(df.columns)-1})")
    print("以降は load_movie_info_embeddings() または add_embedding_features_to_dataframe() で使い回せます。")


if __name__ == "__main__":
    main()
