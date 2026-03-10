#!/usr/bin/env python3
"""
OpenAI API の上位モデル（例: text-embedding-3-large）で
- movie_info（あらすじのみ）
- タイトル + 映画情報
の2種類を一括で embedding して保存するスクリプト。

既存の text-embedding-3-small で作ったファイルは **上書きせず**、
別ファイルとして保存する。

使い方（プロジェクトルートで）:
  python3 archive/run_openai_embeddings_large_models_once.py

必要条件:
  - config/openai_api_key.txt に OpenAI API キーが設定されているか、
    環境変数 OPENAI_API_KEY が設定されていること。
"""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

import lib.openai_embeddings as oe  # type: ignore


def main():
    # ここでのみ「上位モデル」に差し替える（ライブラリ本体は変更しない）
    target_model = "text-embedding-3-large"
    print(f"Using embedding model: {target_model}")
    oe.EMBEDDING_MODEL = target_model

    # 保存先を small 版と分ける
    info_path = Path("outputs/embeddings/movie_info_embeddings_large.pkl")
    title_info_path = Path("outputs/embeddings/movie_title_info_embeddings_large.pkl")

    info_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) movie_info だけ
    print("--- movie_info の embedding (large) を計算します ---")
    df_info = oe.compute_and_save_movie_info_embeddings(
        save_path=info_path, force=True
    )
    print(
        f"movie_info_embeddings_large: {info_path} "
        f"(行数={len(df_info)}, 次元={len(df_info.columns) - 1})"
    )

    # 2) タイトル + 映画情報
    print("--- タイトル + 映画情報 の embedding (large) を計算します ---")
    df_title_info = oe.compute_and_save_title_info_embeddings(
        save_path=title_info_path, force=True
    )
    print(
        f"movie_title_info_embeddings_large: {title_info_path} "
        f"(行数={len(df_title_info)}, 次元={len(df_title_info.columns) - 1})"
    )

    print("完了しました。学習では path= に *_large.pkl を指定して読み込んでください。")


if __name__ == "__main__":
    main()

