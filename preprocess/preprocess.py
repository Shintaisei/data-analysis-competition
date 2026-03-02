"""
データ読み込み。Kaggle とローカルでデータパスを切り替える。
"""
import os
import pandas as pd


def get_data_dir() -> str:
    """コンペデータのディレクトリを返す（Kaggle なら input、そうでなければ data/）。"""
    if os.path.exists("/kaggle/input/matsuo-institute-ds-dojo-4/train.csv"):
        return "/kaggle/input/matsuo-institute-ds-dojo-4"
    return os.path.join(os.getcwd(), "data")


def load_train_test():
    """train.csv と test.csv を読み込み、(train, test) のタプルで返す。"""
    data_dir = get_data_dir()
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))
    return train, test
