"""
Seen/Unseen（批評家基準）の EDA。
train/test の critic_name のみを使い、メモリを抑えて集計する。
"""
import os
import pandas as pd

def main():
    data_dir = "data" if os.path.exists("data/train.csv") else "/kaggle/input/matsuo-institute-ds-dojo-4"
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    # Train: チャンクで読んで批評家の集合と行数を取得
    train_critics = set()
    n_train_rows = 0
    for chunk in pd.read_csv(train_path, usecols=["critic_name"], chunksize=100_000):
        train_critics.update(chunk["critic_name"].astype(str).unique())
        n_train_rows += len(chunk)

    # Test: 1回で読めるサイズ
    test = pd.read_csv(test_path, usecols=["critic_name"])
    test_critics = set(test["critic_name"].astype(str).unique())
    test_in_train = test["critic_name"].astype(str).isin(train_critics)
    n_seen_rows = test_in_train.sum()
    n_unseen_rows = (~test_in_train).sum()

    print("=== Seen/Unseen (by critic) EDA ===")
    print("Train:", n_train_rows, "rows", len(train_critics), "unique critics")
    print("Test:", len(test), "rows", len(test_critics), "unique critics")
    seen_critics = test_critics & train_critics
    unseen_critics = test_critics - train_critics
    print("Test critics in train (seen):", len(seen_critics), f"({100*len(seen_critics)/len(test_critics):.1f}%)")
    print("Test critics NOT in train (unseen):", len(unseen_critics), f"({100*len(unseen_critics)/len(test_critics):.1f}%)")
    print("Test rows with seen critic:", n_seen_rows, f"({100*n_seen_rows/len(test):.1f}%)")
    print("Test rows with unseen critic:", n_unseen_rows, f"({100*n_unseen_rows/len(test):.1f}%)")

if __name__ == "__main__":
    main()
