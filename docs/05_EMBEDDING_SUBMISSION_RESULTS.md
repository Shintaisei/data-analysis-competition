# Embedding 提出結果（Public Score）

embedding 種別 × PCA 次元数の提出比較結果と、**最良設定を基準にした次元削減 8 パターン**の記録。

---

## 1. Embedding 種別 × PCA 8/16 の Public 比較

4 種類の embedding（映画情報のみ / 映画情報 large / タイトル+映画情報 / タイトル+映画情報 large）× PCA 8 次元・16 次元で提出し、Public Score を比較した。

| 提出ファイル | Embedding | PCA | Public Score |
|-------------|-----------|-----|--------------|
| submission_embedding_movie_title_info_pca16.csv | **movie_title_info**（タイトル+映画情報・small） | 16 | **0.75449** |
| submission_embedding_movie_title_info_large_pca16.csv | movie_title_info_large | 16 | 0.75440 |
| submission_embedding_movie_title_info_pca8.csv | movie_title_info（small） | 8 | 0.75437 |
| submission_embedding_movie_title_info_large_pca8.csv | movie_title_info_large | 8 | 0.75388 |
| submission_embedding_movie_info_pca8.csv | movie_info（映画情報のみ） | 8 | 0.75394 |
| submission_embedding_movie_info_large_pca16.csv | movie_info_large | 16 | 0.75370 |
| submission_embedding_movie_info_pca16.csv | movie_info | 16 | 0.75237 |
| submission_embedding_movie_info_large_pca8.csv | movie_info_large | 8 | 0.75317 |

**結論**

- **最高**: **movie_title_info（small）+ PCA 16** → **Public 0.75449**
- 映画情報のみ（movie_info）より、**タイトル＋映画情報の small 版**が Public で最も良い。
- この設定を基準に、次元削減手法を PCA 以外（SVD, ICA, NMF, UMAP 等）に変えた 8 パターンを試す。

---

## 2. 次元削減 8 パターン（movie_title_info 固定）

**基準**: embedding = `movie_title_info`（small）、削減後次元数 = 16。  
**ノートブック**: `train_embedding_submission.ipynb`

| # | 手法 | 説明 | Public Score（記入欄） |
|---|------|------|------------------------|
| 1 | pca | PCA（分散最大化） | **0.75449**（基準・最高） |
| 2 | truncated_svd | TruncatedSVD | **0.75389** |
| 3 | ica | FastICA（独立成分） | — |
| 4 | random_projection | Gaussian Random Projection | — |
| 5 | sparse_pca | Sparse PCA | — |
| 6 | nmf | NMF（非負値化して適用） | — |
| 7 | kernel_pca | Kernel PCA（RBF） | — |
| 8 | umap | UMAP | — |

提出ファイル名: `submission_embedding_movie_title_info_{手法名}{次元数}.csv`  
（例: `submission_embedding_movie_title_info_truncated_svd16.csv`）

### 今回の提出結果（2 本で打ち止め）

| 提出ファイル | Public Score |
|-------------|--------------|
| submission_embedding_movie_title_info_pca16.csv | **0.75449** |
| submission_embedding_movie_title_info_truncated_svd16.csv | 0.75389 |

PCA 16 が TruncatedSVD 16 をわずかに上回り、この 2 本で実験を終了。次元削減の差は出るがスコア差は小さい。なかなか難しいコンペ。

---

## 3. 実験の流れと追加実験（§4）

**ノートブック**: `train_embedding_submission.ipynb`

### 3.1 実行の流れ

1. **§1 設定**: `EMBEDDING_NAME`（最良は `movie_title_info`）、`N_COMPONENTS`（例: 16）、`REDUCTION_METHODS`（8 手法のうち試すもの）、`EXTRA_EXPERIMENTS`（追加実験のキーリスト）。
2. **§2 データ**: `get_baseline_data()` で train/test、embedding を読み込み `E_train` / `E_test` を用意。**§4 だけ回す場合も §2 は必須。**
3. **§3 次元削減 8 パターン**: 各手法で「train だけで fit → train/test を transform → ベースライン特徴量と結合 → LGB 学習 → 予測 → 提出 CSV 保存・検証」。
4. **§4 追加実験**: `EXTRA_EXPERIMENTS` に書いたものだけ実行。提出回数が限られているときに 4 本だけ試す用。

### 3.2 追加実験の 4 パターン

| キー | 内容 | 提出 1 本 |
|------|------|-----------|
| `seed_avg` | movie_title_info + PCA 16 をシード 42, 43, 44 で 3 本学習し、予測を平均。分散低減で微増を狙う。 | ○ |
| `pca24` | movie_title_info + PCA **24** 次元で 1 本。次元を少し増やした効果を確認。 | ○ |
| `ensemble_two` | 既存の `movie_title_info_pca16` と `movie_title_info_large_pca16` の CSV を ID でマージし、target を 0.5:0.5 で平均。2 本の CSV が `outputs/submissions/` にある必要あり。 | ○ |
| `truncated_svd` | movie_title_info + TruncatedSVD 16 で 1 本。PCA 以外の次元削減を 1 本だけ試す用。 | ○ |

### 3.3 技術の詳細

各次元削減手法・追加実験の**特徴・数理・使い分け**は **`docs/06_EMBEDDING_REDUCTION_TECHNOLOGY.md`** にまとめた。コンペで選ぶときの参考にすること。

---

## 4. 関連ファイル・ドキュメント

| ファイル | 役割 |
|----------|------|
| `train_embedding_submission.ipynb` | 最良 embedding（movie_title_info）固定で、次元削減 8 パターンの提出 CSV を一括作成 |
| `lib/embedding_reduction.py` | 次元削減 8 手法の fit/transform ラッパー。出力の NaN/Inf は train の列中央値で補正。 |
| `lib/submission.py` | 提出 CSV の保存（`save_submission`）・検証（`verify_submission`）。予測は [0,1] clip と NaN/Inf→0.5 で sanitize。 |
| `outputs/submissions/` | 提出用 CSV の保存先 |
| **`docs/06_EMBEDDING_REDUCTION_TECHNOLOGY.md`** | 次元削減 8 手法＋追加実験の技術解説（特徴・数理・使い分け） |
