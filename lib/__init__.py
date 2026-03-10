"""
ノートブック用ライブラリ。パイプライン・予測分析・エンコーディング・テキストベクトルをまとめて提供。
"""
from lib.pipeline import get_baseline_data, BASELINE_FEATURES, BASELINE_LGB_PARAMS, prepare_baseline_data
from lib.analysis import add_prediction_analysis, summarize_errors_by, run_full_analysis
from lib.encodings import (
    movie_info_meta,
    ts_te,
    ts_te_binned,
    freq,
    per_movie_ts,
    missing_flags,
    loo,
    add_freq_multi,
    ts_te_multi,
)
from lib.text_vectors import build_vectors, get_available_configs, get_config_descriptions

__all__ = [
    "get_baseline_data",
    "BASELINE_FEATURES",
    "BASELINE_LGB_PARAMS",
    "prepare_baseline_data",
    "add_prediction_analysis",
    "summarize_errors_by",
    "run_full_analysis",
    "movie_info_meta",
    "ts_te",
    "ts_te_binned",
    "freq",
    "per_movie_ts",
    "missing_flags",
    "loo",
    "add_freq_multi",
    "ts_te_multi",
    "build_vectors",
    "get_available_configs",
    "get_config_descriptions",
]
