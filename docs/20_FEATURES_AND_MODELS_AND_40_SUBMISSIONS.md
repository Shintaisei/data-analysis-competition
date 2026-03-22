# 現状の「特徴量 vs モデル」と残り 40 回の使い方

「今は同じ特徴量をベースにモデルだけ違う」という認識は **半分だけ正しい**。実際は **成分ごとに特徴量もモデルもバラバラ**。ここで整理する。

---

## 0. 「55 特徴」の中身

`use_best_pipeline=True` のとき、**55 特徴**は次の 3 ブロックで構成される（`lib/improvement_candidates.py` の `_build_best_pipeline_data` ＋ `lib/pipeline.py` の `BASELINE_FEATURES`）。

| ブロック | 個数 | 中身 |
|----------|------|------|
| **ベースライン（3C3＋テキストメタ）** | **38** | 下記の通り。 |
| **埋め込み PCA** | **16** | `movie_title_info` の embedding を PCA で 16 次元に圧縮した列（`pca_0` 〜 `pca_15`）。 |
| **交差 1 列** | **1** | `critic_te_x_genre_Documentary`（批評家 TE × ジャンル Documentary）。 |

**合計**: 38 + 16 + 1 = **55**。

### 38 の内訳（ベースライン）

- **ID・メタ**: `rotten_tomatoes_link`, `critic_name`, `top_critic`, `publisher_name`, `movie_title`, `movie_info`, `content_rating`, `directors`, `authors`, `actors`, `runtime`, `production_company`
- **日付・経過**: `review_year`, `review_month`, `review_dayofweek`, `release_year`, `release_month`, `release_dayofweek`, `movie_age_days`
- **ジャンル（0/1）**: `genre_Drama`, `genre_Comedy`, `genre_Action`, `genre_Mystery`, `genre_Fantasy`, `genre_Romance`, `genre_Horror`, `genre_Documentary`
- **Target Encoding 等**: `critic_name_te_ts`, `production_company_te_ts`, `critic_name_te_ts_bin`, `production_company_te_ts_bin`
- **ビン・メタ**: `runtime_bin`, `movie_age_bin`, `release_decade`, `movie_info_len`, `movie_info_word_count`, `movie_title_len`, `movie_title_word_count`

※ `improvement_05`（55 のみ）はこの **55 列だけ**で、BPR や 2-hop は使っていない。

---

## 1. 現最高の 4 本は「同じ特徴・モデルだけ違う」か？ → **違う**

現最高（0.76967）は **4 本の提出のブレンド**。各 1 本の中身は次のとおり。

| 1 本目 | 提出ファイルの元 | 特徴量 | モデル |
|--------|------------------|--------|--------|
| **count1** | submission_2hop_bpr64_count1.csv | **55 特徴 + BPR64 埋め込み + movie_review_count（2-hop の 1 列）** | LGB のみ |
| **BPR128** | submission_2hop_bpr128_only.csv | **55 特徴 + BPR128 埋め込み**（2-hop なし） | LGB のみ |
| **stacking** | submission_improvement_03_stacking.csv | **count1 と同一**（55 + BPR64 + movie_review_count） | **LGB + XGB + CatBoost → Ridge**（3 モデル + メタ） |
| **4 本目** | similar / improvement_05 / 2 本ブレンド | **パターンで違う**（下表） | LGB または「予測のブレンド」 |

### 4 本目のパターン別

| 4 本目の中身 | 特徴量 | モデル |
|--------------|--------|--------|
| similar_movies_reviewed | 55 + BPR64 + **similar_movies_reviewed_count**（1 列） | LGB |
| improvement_05 | **55 特徴のみ**（BPR なし） | LGB（scale_pos_weight 付き） |
| 2 本ブレンド（count1+BPR128） | 予測だけの合成なので「特徴」はなし | 上記 count1 と BPR128 の加重平均 |

### まとめ：何が同じで何が違うか

- **特徴量が同じなのは「count1 と stacking」だけ**。この 2 本は **同じ特徴でモデルだけ違う**（1 本は LGB、1 本は LGB+XGB+CB→Ridge）。
- それ以外は **特徴量が違う**：
  - BPR128 は「BPR64 → BPR128」「2-hop なし」で別構成。
  - 4 本目は「55 のみ」「55+BPR64+類似 1 列」や「2 本の予測ブレンド」など、さらに別。

だから **「同じ特徴量をベースにモデルだけ違う」は count1 と stacking の関係に限った話**で、4 本全体でいうと **特徴もモデルも混在** している。

---

## 2. 残り 40 回でやること：「モデルだけ」ではない

残り 40 回は **「モデルをひたすら変えて試す」だけではない**。次の 2 軸がある。

| 軸 | 意味 | 例 |
|----|------|-----|
| **特徴量を変える** | 今の 55+BPR+2hop に **1 列足す** or **別の特徴セットで 1 本作る** | メタ 2-hop、類似映画の別定義、NMF、TF-IDF SVD、Surprise を 1 特徴に |
| **モデルを変える** | **同じ特徴**のまま学習器だけ変える | 同じ 55+BPR64+count1 で NN、重い LGB、別 seed のスタッキング |

- **特徴を変える** → 新しい「1 本」が増える。それを既存 4 本とブレンド（5 本目として重みを振る）などで試す。
- **モデルを変える** → 既存の「1 本」を別モデルに差し替えたバージョンを作り、同じ 4 本ブレンドの構成で重みを変えたりして試す。

両方やれるので、「ひたすらモデルだけ」に絞る必要はない。

---

## 3. 残り 40 回の使い方の例（優先度イメージ）

| 優先度 | やること | 特徴量 | モデル | 提出の使い方 |
|--------|----------|--------|--------|--------------|
| 高 | 特徴を 1 列足して 1 本作る | 55+BPR64+count1 + **メタ 2-hop** or **NMF** or **類似の別定義** | LGB or スタッキング | その 1 本を 5 本目として 0.05〜0.15 でブレンド（複数重みで数回） |
| 高 | 別モデルを 1 本作る | 55+BPR64+count1（**同じ**） | **NN** or **重い LGB** or **Surprise を 1 特徴にした LGB** | 同上、5 本目としてブレンド |
| 中 | スタッキングの seed を変える | count1 と同一 | LGB+XGB+CB→Ridge（**seed だけ別**） | その 1 本で stacking を差し替え、4 本ブレンドで数パターン |
| 中 | 4 本ブレンドの重みを変える | 変更なし（既存 CSV のみ） | 変更なし | 新モデルなしで 0.20:0.20:0.45:0.15 の周辺を数回 |
| 低 | GNN / LightGCN | 接続情報中心 | GNN | 1 本出して 4 本 or 5 本ブレンドに組み込む |

「モデルひたすら変える」に寄せるなら、**同じ特徴で NN・重い LGB・Surprise 入り LGB などを 1 本ずつ作り、それぞれ 5 本目としてブレンド** する形になる。その場合でも、**特徴を 1 列足した 1 本** を混ぜるパターンと組み合わせると、40 回を特徴軸とモデル軸の両方で使いやすい。

---

## 4. 一言まとめ

- **今の 4 本は「同じ特徴・モデルだけ違う」ではない。** count1 と stacking だけ特徴が同じで、他は特徴もモデルも違う。
- **残り 40 回は「モデルだけ変えて試す」に限定しなくてよい。** 特徴を 1 列足した 1 本を作る試行と、同じ特徴でモデルだけ変えた 1 本を作る試行の、両方を並行してよい。
- 運用の目安：**新しく「1 本」を作るたびに、それを 5 本目にしたブレンドを 2〜3 重みで出し、効いたらその軸（特徴 or モデル）を少し広げる** と、40 回を整理して使える。
