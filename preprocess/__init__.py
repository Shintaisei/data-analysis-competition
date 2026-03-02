"""
前処理: データ読み込み。
Kaggle / ローカル両対応のパス解決と train / test の読み込み。
"""
from .preprocess import get_data_dir, load_train_test

__all__ = ["get_data_dir", "load_train_test"]
