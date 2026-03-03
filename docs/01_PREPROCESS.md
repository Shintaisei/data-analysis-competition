# 前処理編 — 何の特徴量に何をしてどうなったか

時系列 CV（val years = 2013〜2016）を共通検証として、**対象カラム・やったこと・結果（CV_AUC・ベース比）** を一覧化する。  
「置き換え」は悪化し、「既存を残して足す」が効いた実験だけ伸びている。

---

## 1. ベースの構成

| 項目 | 内容 |
|------|------|
| **特徴数** | 34 列（3C3 込みで運用時は 38 列＋テキストメタ 4 列など）。 |
| **カテゴリ 8 列** | critic_name, production_company, directors, authors, rotten_tomatoes_link, movie_title, movie_info, actors → `fillna("missing").astype("category")` のまま渡す。 |
| **数値・日付** | runtime（欠損は中央値）, movie_age_days, release_year/month/dayofweek, review_year/month/dayofweek, top_critic。 |
| **その他** | ジャンル One-hot 8 列（genre_Drama 等）, publisher_name, content_rating（category）。 |
| **CV_AUC 目安** | **0.7600** 前後（3C3 込み）。3C3 なしベースは **0.7369** 前後。 |

---

## 2. 成功した実験（CV 改善）

| 対象カラム | やったこと | 結果（CV_AUC / ベース比） |
|------------|------------|---------------------------|
| **critic_name, production_company** | category は**残したまま**、時系列 TE（その行より前の同カテゴリ Fresh 率、m=10）を**追加** → `critic_name_te_ts`, `production_company_te_ts` | 0.7425 / **+0.0056**（ベース 0.7369） |
| **上記 2 列** | 上記に加え、TE を 3 ビン（[0,0.33)→0, [0.33,0.67)→1, [0.67,1.01]→2）に切った列を**追加** → `*_te_ts_bin` | 0.7426 / **+0.0057** |
| **critic_name, production_company, runtime, movie_age_days, release_year** | 時系列 TE 追加に加え、runtime_bin, movie_age_bin, release_decade を**追加**（元の数値列は残す） | 0.7425 / **+0.0056** |

**ポイント**: いずれも「既存列は残し、新列だけ足す」。批評家・制作会社の**時系列 TE 追加**が最も効いた。

---

## 3. 微増した実験

| 対象カラム | やったこと | 結果（CV_AUC / ベース比） |
|------------|------------|---------------------------|
| **movie_info** | category を**外し**、文字数・語数の 2 列に**置き換え**（テキスト中身は失う） | 0.7369 / +0.0012（環境差あり） |
| **publisher_name** | category は残したまま、出現回数 `publisher_name_freq` を**追加** | 0.7603 / +0.0003 |
| **movie_info** | category は**残したまま**、文字数・語数 `movie_info_len`, `movie_info_word_count` を**追加**（movie_info_meta） | 0.7602 / +0.0003 |
| **movie_info, movie_title** | 上記に加え、`movie_title_len`, `movie_title_word_count` を**追加**（movie_info_and_title_meta） | 0.76022 / +0.00026（**採用・提出済み**） |
| **movie_info** | TF-IDF(200) → TruncatedSVD(20) で 20 列を**追加**（tr のみ fit） | 0.76012 / +0.00016 |

---

## 4. 変化なしの実験

| 対象カラム | やったこと | 結果 |
|------------|------------|------|
| critic_name | 頻度を追加 | 0.7356 / 0.0000 |
| runtime, movie_age_days, release_year | ビンのみ追加（ts_te なし） | 0.7369 / 0.0000 |
| runtime, movie_age_days | 欠損フラグ追加 | 0.7369 または 0.7600 / 0.0000 |
| 複数カテゴリ列 | 頻度を一括追加 | 0.7600 / 0.0000 |
| movie_title | 文字数・語数を 1 列ずつ追加 | 0.75996 / 0.00000 |

---

## 5. 失敗した実験（CV 悪化）— 繰り返さない

| 対象カラム | やったこと | 結果（CV_AUC / ベース比） | 主な原因 |
|------------|------------|---------------------------|----------|
| **8 列（critic_name, production_company, directors, authors, rotten_tomatoes_link, movie_title, movie_info, actors）** | 全列を「全体の Fresh 率」に**置き換え**（category 削除） | 0.7030 / **-0.034** | 置き換えで木のグルーピングを捨てた |
| **上記 8 列** | 全列を出現回数に**置き換え**（category 削除） | 0.6920 / **-0.045** | 頻度は target と無関係で情報が弱い |
| **critic_name** | LOO（自分除く同カテゴリ target 平均）を**追加** | 0.7203 / -0.0166 | リークに近くノイズ |
| **directors** | LOO を追加 | 0.7317 / -0.0282 | 同上 |
| **authors** | LOO を追加 | 0.7062 / **-0.0538** | カーディナリティ高く不安定 |
| **directors** | 時系列 TE を追加 | 0.7471 / -0.0129 | 監督は作品ごとに評価が変わり過去 Fresh 率が効かない |
| **directors** | 時系列 TE ＋ 3 ビン化を追加 | 0.7470 / -0.0130 | 同上 |
| **authors** | 時系列 TE を追加 | 0.7385 / -0.0215 | カーディナリティ高く不安定 |
| **authors** | 時系列 TE ＋ 3 ビン化を追加 | 0.7385 / -0.0215 | 同上 |
| **複数カテゴリ列** | 時系列 TE を一括追加（directors/authors 含む） | 0.7429 / -0.0171 | 効かない列を含むと全体悪化 |
| **rotten_tomatoes_link** | 映画ごと時系列レビュー数・Fresh 率を追加 | 0.7423 / -0.0177 | 映画ごと集計がノイズに |
| **directors** | 頻度を追加 | 0.7277 / -0.0080 | 出現回数は Fresh と無関係 |
| **authors** | 頻度を追加 | 0.7291 / -0.0065 | 同上 |
| **rotten_tomatoes_link** | 頻度を追加 | 0.7546 / -0.0054 | レビュー数は他特徴で十分 |
| **movie_title** | 頻度を追加 | 0.7551 / -0.0049 | ほぼユニークで情報なし |
| **actors** | 頻度を追加 | 0.7562 / -0.0038 | 出演回数は Fresh と無関係 |
| **production_company** | 頻度を追加 | 0.7598 / -0.0001 | ほぼ変化なし |
| **movie_title** | category を**外し**頻度のみに**置き換え** | 0.7339 等 / -0.0018〜-0.0043 | タイトル情報を捨てた |
| **movie_info** | category を外し長さ・語数のみに**置き換え**（別実験） | 0.7365 / -0.0004 | 置き換えで category を捨てた |
| **movie_info, movie_title, runtime 等** | 上記置き換え＋ビンの組み合わせ | 0.7350 / -0.0019 | 失敗した置き換えの重ねがけ |

---

## 6. テキストベクトル化（movie_info 等）

**入力**: movie_info（あらすじ）。**共通**: 各 fold の tr のみで fit、val/te は transform のみ。ベース CV_AUC = **0.7600**（34 特徴）。  
**結論**: いずれも **+0.0001〜0.0006** 程度の微増。あらすじ内容と Fresh の相関は弱い。

### 6.1 記録済み結果（train_text_vector_experiments / lib.text_vectors）

| 対象 | やったこと（処理） | 結果（CV_AUC / ベース比） |
|------|-------------------|---------------------------|
| なし（base） | テキストベクトルなし | 0.7600 / 0.0000 |
| movie_info | TfidfVectorizer(50) で 50 次元追加 | 0.7600 / 0.0000 |
| movie_info | TfidfVectorizer(100) で 100 次元追加 | 0.7600 / 0.0000 |
| movie_info | TF-IDF(200) → TruncatedSVD(20) で 20 次元追加 | 0.7601 / +0.0001 |
| movie_info | CountVectorizer(50) で 50 次元追加（BoW） | 0.7603 / +0.0003 |
| movie_info | Count(200) → SVD(20) で 20 次元追加 | 0.7603 / +0.0003 |
| movie_info | HashingVectorizer 先頭 50 次元追加 | 0.7600 / 0.0000 |
| movie_info | Hash(512) → SVD(20) で 20 次元追加 | 0.7601 / +0.0001 |
| movie_info | TF-IDF(200) → LDA(10) で 10 次元追加 | 0.7606 / **+0.0006** |
| movie_info | TF-IDF(200) → LDA(20) で 20 次元追加 | 0.7603 / +0.0003 |
| movie_info | TF-IDF(200) → NMF(10) で 10 次元追加 | 0.7600 / 0.0000 |
| movie_info | NMF(20) | 未記録（実行途中停止） |

### 6.2 登録済み config の処理内容（アルゴリズム）

| config | 入力 | 処理内容 |
|--------|------|----------|
| tfidf_50 / tfidf_100 | movie_info | TfidfVectorizer(max_features=50 or 100, min_df=2, max_df=0.95, ngram=(1,2), sublinear_tf) |
| tfidf_svd20 | movie_info | TF-IDF(200) → TruncatedSVD(20) |
| count_50 | movie_info | CountVectorizer(50, min_df=2, max_df=0.95, ngram=(1,2)) |
| count_svd20 | movie_info | Count(200) → TruncatedSVD(20) |
| hash_50 | movie_info | HashingVectorizer(256, ngram=(1,2), norm="l2") の先頭 50 次元 |
| hash_svd20 | movie_info | Hash(512) → TruncatedSVD(20) |
| lda_10 / lda_20 | movie_info | TF-IDF(200) → LDA(10 or 20, max_iter=10) |
| nmf_10 / nmf_20 | movie_info | TF-IDF(200) → NMF(10 or 20, max_iter=100) |
| concat_mi_title_* | movie_title + " . " + movie_info | 結合テキストを TF-IDF / Count 等でベクトル化（プレフィックス ct_*） |
| doc2vec_32 | movie_info | Doc2Vec(32, window=5, min_count=2, epochs=5) |
| word2vec_32 | movie_info | Word2Vec 学習後、文は単語ベクトル平均で 32 次元 |
| sentence_transformer_32 | 事前計算埋め込み | 先頭 32 次元を使用 |

**再開**: nmf_20 以降をやりたい場合は、ノートの「再開用」セルで `RECORDED_PARTIAL_RESULTS` と `START_FROM_CONFIG = "nmf_20"` を設定して CV セルを実行。

---

## 7. 技術メモ（アルゴリズム要約）

| 技術 | やること | 備考 |
|------|----------|------|
| **時系列 TE（ts_te）** | その行より前の同カテゴリの Fresh 率をスムージング（m=10）して 1 列追加。val/te は tr のカテゴリ別スムージング平均でマッピング。 | critic_name, production_company で有効。directors/authors では悪化。 |
| **時系列 TE の 3 ビン化** | TE 値を [0,0.33)→0, [0.33,0.67)→1, [0.67,1.01]→2 に切った列を追加。 | 単体では ts_te と同程度。 |
| **頻度** | 学習データ内の出現回数（整数）を追加。target は使わない。 | 多くのカテゴリで無効または悪化。publisher_name のみ微増。 |
| **LOO** | 自分を除いた同カテゴリの target 平均（スムージングあり）を追加。 | 全列で悪化。リークに近い。 |
| **ビニング** | runtime_bin（[0,90,120,150,1000]）, movie_age_bin（日数で 5 段）, release_decade = (year//10)*10。元の数値は残す。 | ts_te と組み合わせると効く。単体では変化なし。 |
| **テキスト長・語数** | movie_info_len, movie_info_word_count, movie_title_len, movie_title_word_count を追加。元の category は残す。 | 置き換えだと悪化。追加なら微増。 |
| **Target Encoding（一括置き換え）** | カテゴリを全体 Fresh 率に**置き換え**（列を消す）。 | 8 列一括で -0.034。**使わない**。 |

---

## 8. 参照ノート・ファイル

| 内容 | ファイル |
|------|----------|
| 前処理比較（時系列 TE・LOO・ビン・頻度） | train_preprocess_experiments.ipynb |
| 一括置き換え（TE・頻度）比較 | train_preprocess_compare.ipynb |
| 1 種類ずつ前処理変更 | train_extended.ipynb |
| ベース＋1つ追加（重要度ベース） | train_importance_experiments.ipynb |
| テキスト（TF-IDF・SVD・メタ） | train_text_experiments.ipynb |
| テキストベクトル化網羅 | train_text_vector_experiments.ipynb, lib.text_vectors |
| 共通エンコーディング | lib.encodings |

**続き**: 特徴量の掛け合わせ・交互作用は **docs/02_FEATURE_ENGINEERING.md**（特徴量エンジニアリング編）を参照。
