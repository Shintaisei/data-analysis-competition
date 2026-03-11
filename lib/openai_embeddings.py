"""
OpenAI Embedding API で movie_info を映画単位で1回だけベクトル化し、保存・再利用する。

- 1回だけ実行: compute_and_save_movie_info_embeddings(force=True) または run_openai_embeddings_once.py
- 以降: load_movie_info_embeddings() で保存済みを読み、train/test にマージして使い回す。
- API キーは環境変数 OPENAI_API_KEY に設定する（コードには書かない）。
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

# 保存先（1回だけ計算した embedding）
DEFAULT_EMBEDDINGS_DIR = Path("outputs/embeddings")
DEFAULT_EMBEDDINGS_PATH = DEFAULT_EMBEDDINGS_DIR / "movie_info_embeddings.parquet"
# タイトル + 映画情報用（別ファイルで保存）
DEFAULT_TITLE_INFO_EMBEDDINGS_PATH = DEFAULT_EMBEDDINGS_DIR / "movie_title_info_embeddings.parquet"

# API キーを貼るファイル（プロジェクトルートからの相対）
CONFIG_KEY_FILE = Path("config/openai_api_key.txt")
CONFIG_KEY_FILE_EXAMPLE = Path("config/openai_api_key.example.txt")


def _project_root() -> Path:
    """プロジェクトルート（lib の親）を返す。"""
    return Path(__file__).resolve().parent.parent


def _load_api_key_from_file() -> bool:
    """
    config/openai_api_key.txt から API キーを読み、環境変数にセットする。
    ファイルが無い場合は .example をコピーして False を返す。
    キーが読み取れたら True。
    """
    if os.environ.get("OPENAI_API_KEY", "").strip():
        return True
    root = _project_root()
    key_file = root / CONFIG_KEY_FILE
    example_file = root / CONFIG_KEY_FILE_EXAMPLE
    if not key_file.exists():
        if example_file.exists():
            key_file.parent.mkdir(parents=True, exist_ok=True)
            key_file.write_text(example_file.read_text(encoding="utf-8"), encoding="utf-8")
            raise RuntimeError(
                f"API キー用のファイルを作成しました: {key_file}\n"
                "  このファイルを開き、1行目に OpenAI の API キー（sk- で始まる）を貼り付けて保存し、もう一度実行してください。"
            )
        raise RuntimeError(
            f"API キーを貼るファイルがありません: {key_file}\n"
            f"  {key_file} を開き、1行目に OpenAI の API キー（sk- で始まる）を貼り付けて保存してから、もう一度実行してください。"
        )
    raw = key_file.read_text(encoding="utf-8").strip()
    # sk- で始まる行をキーとして使う（複数行の場合は最初の有効行）
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("sk-") and len(line) > 20:
            os.environ["OPENAI_API_KEY"] = line
            return True
    raise RuntimeError(
        f"{key_file} に有効な API キーがありません。\n"
        "  1行目に sk- で始まる OpenAI の API キーを貼り付けて保存してください。"
    )

# モデル・バッチ
EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 200  # 1リクエストあたりのテキスト数（APIは最大2048まで可。長文のためトークン上限を考慮）

# 料金目安（text-embedding-3-small, 2024年頃: $0.00002 / 1K tokens）
# 最新: https://openai.com/api/pricing/ または https://platform.openai.com/docs/models/text-embedding-3-small
EMBEDDING_PRICE_PER_1K_TOKENS_USD = 0.00002


def _estimate_tokens(texts: list[str]) -> int:
    """トークン数を概算する。tiktoken があればそれを使い、なければ文字数/2 で保守的に見積もる。"""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return sum(len(enc.encode(t)) for t in texts)
    except Exception:
        return sum(max(1, len(t) // 2) for t in texts)


def estimate_embedding_cost(texts: Optional[list[str]] = None) -> dict:
    """
    ［API は叩かず］対象映画数・推定トークン数・推定料金（USD）を返す。
    texts を渡さない場合は movie_info のみのテキストで見積もる。
    """
    if texts is None:
        unique = _get_unique_movies_with_info()
        texts = unique["movie_info"].tolist()
    return _estimate_embedding_cost_from_texts(texts)


def _estimate_embedding_cost_from_texts(texts: list[str]) -> dict:
    n_texts = len(texts)
    estimated_tokens = _estimate_tokens(texts)
    estimated_usd = (estimated_tokens / 1000.0) * EMBEDDING_PRICE_PER_1K_TOKENS_USD
    n_batches = (n_texts + BATCH_SIZE - 1) // BATCH_SIZE
    return {
        "n_movies": n_texts,
        "estimated_tokens": estimated_tokens,
        "estimated_usd": round(estimated_usd, 6),
        "n_api_calls": n_batches,
        "price_per_1k_tokens_usd": EMBEDDING_PRICE_PER_1K_TOKENS_USD,
    }


def _get_unique_movies_with_info():
    """train + test から映画単位でユニークな (rotten_tomatoes_link, movie_info) を返す。"""
    from preprocess import load_train_test

    train, test = load_train_test()
    train_movies = (
        train[["rotten_tomatoes_link", "movie_info"]]
        .drop_duplicates(subset=["rotten_tomatoes_link"])
        .copy()
    )
    test_movies = (
        test[["rotten_tomatoes_link", "movie_info"]]
        .drop_duplicates(subset=["rotten_tomatoes_link"])
        .copy()
    )
    # train を優先し、test で train にない link だけ追加
    train_links = set(train_movies["rotten_tomatoes_link"])
    test_only = test_movies[~test_movies["rotten_tomatoes_link"].isin(train_links)]
    unique = pd.concat([train_movies, test_only], ignore_index=True)
    unique["movie_info"] = unique["movie_info"].fillna("").astype(str)
    return unique


def _get_unique_movies_title_and_info():
    """train + test から映画単位でユニークな (rotten_tomatoes_link, タイトル+映画情報の文字列) を返す。"""
    from preprocess import load_train_test

    train, test = load_train_test()
    for df in (train, test):
        if "movie_title" not in df.columns:
            raise ValueError("movie_title 列が必要です。load_train_test() の CSV に含まれているか確認してください。")
    train_movies = (
        train[["rotten_tomatoes_link", "movie_title", "movie_info"]]
        .drop_duplicates(subset=["rotten_tomatoes_link"])
        .copy()
    )
    test_movies = (
        test[["rotten_tomatoes_link", "movie_title", "movie_info"]]
        .drop_duplicates(subset=["rotten_tomatoes_link"])
        .copy()
    )
    train_links = set(train_movies["rotten_tomatoes_link"])
    test_only = test_movies[~test_movies["rotten_tomatoes_link"].isin(train_links)]
    unique = pd.concat([train_movies, test_only], ignore_index=True)
    title = unique["movie_title"].fillna("").astype(str)
    info = unique["movie_info"].fillna("").astype(str)
    unique["text"] = "Title: " + title + ". " + info
    return unique[["rotten_tomatoes_link", "text"]]


def estimate_title_info_embedding_cost() -> dict:
    """［API は叩かず］タイトル+映画情報で embedding する場合のコスト見積もり。"""
    unique = _get_unique_movies_title_and_info()
    return _estimate_embedding_cost_from_texts(unique["text"].tolist())


def _call_openai_embeddings_batch(texts: list[str], model: str = EMBEDDING_MODEL):
    """OpenAI Embeddings API を1バッチ分呼ぶ。texts は長さ BATCH_SIZE 以下を想定。"""
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    # 空文字は API がエラーにする場合があるので短いプレースホルダに
    inputs = [t if t.strip() else "(no description)" for t in texts]
    resp = client.embeddings.create(input=inputs, model=model)
    return [e.embedding for e in resp.data]


def compute_and_save_movie_info_embeddings(
    save_path: Optional[Union[Path, str]] = None,
    force: bool = False,
) -> pd.DataFrame:
    """
    映画単位で movie_info を OpenAI でベクトル化し、parquet で保存する。
    既に save_path が存在し force=False なら保存済みを読み込んで返す。

    Returns
    -------
    DataFrame
        columns: rotten_tomatoes_link, emb_0, emb_1, ... (次元はモデル依存)
    """
    save_path = Path(save_path) if save_path else DEFAULT_EMBEDDINGS_PATH
    if save_path.exists() and not force:
        return pd.read_parquet(save_path)

    # 環境変数になければ config/openai_api_key.txt から読み込む
    _load_api_key_from_file()
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        raise RuntimeError(
            "OPENAI_API_KEY が設定されていません。環境変数か config/openai_api_key.txt に設定してください。"
        )

    unique = _get_unique_movies_with_info()
    link_col = unique["rotten_tomatoes_link"]
    texts = unique["movie_info"].tolist()
    n = len(texts)

    all_embeddings = []
    for start in range(0, n, BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        emb = _call_openai_embeddings_batch(batch)
        all_embeddings.extend(emb)
        if start + BATCH_SIZE < n:
            time.sleep(0.2)  # レート制限対策

    arr = np.array(all_embeddings, dtype=np.float32)
    dim = arr.shape[1]
    out = pd.DataFrame(
        arr,
        columns=[f"emb_{i}" for i in range(dim)],
        index=unique.index,
    )
    out.insert(0, "rotten_tomatoes_link", link_col)
    out = out.reset_index(drop=True)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        out.to_parquet(save_path, index=False)
    except ImportError:
        pkl_path = save_path.with_suffix(".pkl")
        out.to_pickle(pkl_path)
    return out


def load_movie_info_embeddings(
    path: Optional[Union[Path, str]] = None,
) -> pd.DataFrame:
    """
    保存済みの movie_info embedding を読み込む。
    ファイルが無い場合は None を返す（または raise するかは呼び出し側で判断）。

    Returns
    -------
    DataFrame
        columns: rotten_tomatoes_link, emb_0, emb_1, ...
    """
    path = Path(path) if path else DEFAULT_EMBEDDINGS_PATH
    pkl_path = path.with_suffix(".pkl")
    if path.suffix.lower() == ".pkl" and path.exists():
        return pd.read_pickle(path)
    if path.exists():
        try:
            return pd.read_parquet(path)
        except ImportError:
            if pkl_path.exists():
                return pd.read_pickle(pkl_path)
            raise
    if pkl_path.exists():
        return pd.read_pickle(pkl_path)
    raise FileNotFoundError(
        f"Embedding ファイルが見つかりません: {path} または {pkl_path}\n"
        "先に compute_and_save_movie_info_embeddings(force=True) または "
        "run_openai_embeddings_once.py を1回実行してください。"
    )


def compute_and_save_title_info_embeddings(
    save_path: Optional[Union[Path, str]] = None,
    force: bool = False,
) -> pd.DataFrame:
    """
    映画単位で「タイトル + movie_info」を連結したテキストを OpenAI でベクトル化し、保存する。
    既に save_path が存在し force=False なら保存済みを読み込んで返す。

    Returns
    -------
    DataFrame
        columns: rotten_tomatoes_link, emb_0, emb_1, ...
    """
    save_path = Path(save_path) if save_path else DEFAULT_TITLE_INFO_EMBEDDINGS_PATH
    if not force:
        pkl_path = save_path.with_suffix(".pkl")
        if save_path.suffix.lower() == ".pkl" and save_path.exists():
            return pd.read_pickle(save_path)
        if save_path.exists():
            try:
                return pd.read_parquet(save_path)
            except ImportError:
                if pkl_path.exists():
                    return pd.read_pickle(pkl_path)
        elif pkl_path.exists():
            return pd.read_pickle(pkl_path)

    _load_api_key_from_file()
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        raise RuntimeError(
            "OPENAI_API_KEY が設定されていません。環境変数か config/openai_api_key.txt に設定してください。"
        )

    unique = _get_unique_movies_title_and_info()
    link_col = unique["rotten_tomatoes_link"]
    texts = unique["text"].tolist()
    n = len(texts)

    all_embeddings = []
    for start in range(0, n, BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        emb = _call_openai_embeddings_batch(batch)
        all_embeddings.extend(emb)
        if start + BATCH_SIZE < n:
            time.sleep(0.2)

    arr = np.array(all_embeddings, dtype=np.float32)
    dim = arr.shape[1]
    out = pd.DataFrame(
        arr,
        columns=[f"emb_{i}" for i in range(dim)],
        index=unique.index,
    )
    out.insert(0, "rotten_tomatoes_link", link_col)
    out = out.reset_index(drop=True)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        out.to_parquet(save_path, index=False)
    except ImportError:
        pkl_path = save_path.with_suffix(".pkl")
        out.to_pickle(pkl_path)
    return out


def load_movie_title_info_embeddings(
    path: Optional[Union[Path, str]] = None,
) -> pd.DataFrame:
    """保存済みの「タイトル+映画情報」embedding を読み込む。"""
    path = Path(path) if path else DEFAULT_TITLE_INFO_EMBEDDINGS_PATH
    pkl_path = path.with_suffix(".pkl")
    if path.suffix.lower() == ".pkl" and path.exists():
        return pd.read_pickle(path)
    if path.exists():
        try:
            return pd.read_parquet(path)
        except ImportError:
            if pkl_path.exists():
                return pd.read_pickle(pkl_path)
            raise
    if pkl_path.exists():
        return pd.read_pickle(pkl_path)
    raise FileNotFoundError(
        f"Embedding ファイルが見つかりません: {path} または {pkl_path}\n"
        "先に compute_and_save_title_info_embeddings(force=True) または "
        "archive/run_openai_embeddings_title_info_once.py を1回実行してください。"
    )


def add_embedding_features_to_dataframe(
    df: pd.DataFrame,
    embeddings: Optional[pd.DataFrame] = None,
    path: Optional[Union[Path, str]] = None,
) -> pd.DataFrame:
    """
    train / test に embedding 列をマージして返す（元の df は変更しない）。

    Parameters
    ----------
    df : DataFrame
        rotten_tomatoes_link 列がある train または test
    embeddings : DataFrame, optional
        すでに load_movie_info_embeddings() で読んだ DataFrame。指定しない場合は path から読む。
    path : Path or str, optional
        embedding の parquet パス（embeddings 未指定時のみ使用）

    Returns
    -------
    DataFrame
        df に emb_0, emb_1, ... をマージしたコピー。対応する link が無い行は NaN。
    """
    if embeddings is None:
        embeddings = load_movie_info_embeddings(path=path)
    emb_cols = [c for c in embeddings.columns if c != "rotten_tomatoes_link"]
    merged = df.merge(
        embeddings[["rotten_tomatoes_link"] + emb_cols],
        on="rotten_tomatoes_link",
        how="left",
        suffixes=("", "_emb"),
    )
    return merged
