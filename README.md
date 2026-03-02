# ds_dojo4 - Rotten Tomatoes Review Prediction

Kaggle から移行したベースライン用のローカル環境です。

## フォルダ構成

```
ds_dojo4/
├── data/                    # データ (train.csv, test.csv)
├── preprocess/              # 前処理（データ読み込み）
│   ├── __init__.py
│   └── preprocess.py        # get_data_dir(), load_train_test()
├── feature_engineering/     # 特徴量エンジニアリング
│   ├── __init__.py
│   └── features.py          # create_features(), FEATURES（ホストベースライン同一）
├── train_baseline.ipynb         # ベースライン（手付かずで戻れる基準）
├── train_preprocess_compare.ipynb  # 前処理強化の比較（ベース vs TE あり）
├── train_extended.ipynb         # 拡張用: 未使用列を追加しベース vs 拡張のスコア比較
├── dojo4-host-baseline-v1-729d27.ipynb  # ホスト配布ベースライン（参照用）
├── requirements.txt
└── README.md
```

## 環境のセットアップ（Kaggle 互換 .venv）

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m ipykernel install --prefix .venv --name ds_dojo4 --display-name "Python 3 (ds_dojo4 .venv)"
```

（すでに実行済みの場合は不要です。）

## ノートブックの実行

1. カーネルを **「Python 3 (ds_dojo4 .venv)」** または **「.venv」** に設定
2. 実行するノートブックを開く
   - **`train_baseline.ipynb`**: 前処理・特徴量は `preprocess` / `feature_engineering` を利用。時系列CVで学習し提出用 CSV を出力する。**いじらず基準として残す。**
   - **`train_preprocess_compare.ipynb`**: train_baseline を複製。前処理が粗い 8 列に Target Encoding を追加し、「ベース（現状前処理のみ）」vs「強化（TE あり）」の CV AUC を比較。前処理でどれだけスコアが上がるか確認する。
   - **`train_extended.ipynb`**: 上を複製した拡張用。未使用列を処理して追加し、「ベース（FEATURESのみ）」と「拡張（ベース+未使用列）」の CV AUC を比較。新しい特徴量・前処理を試すときはこのファイルで行う。

データは `data/` フォルダを参照しています。
