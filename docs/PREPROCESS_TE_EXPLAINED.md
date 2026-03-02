# 前処理・特徴量実験の全記録

**構成**
- **1.** ベース（34 特徴の構成）
- **2.** 成功した実験（CV 改善）
- **3.** 微増した実験
- **4.** 変化なしの実験
- **5.** 失敗した実験（CV 悪化）
- **6.** テキストベクトル化の網羅実験（途中）
- **7.** 技術別アルゴリズムと値の変化
- **8.** 用語・参照

各実験は同じフォーマット（対象カラム / やったこと / 元の値 → 処理後の値 / CV_AUC / 差分 / 出典）で記載。ベースの CV_AUC はノートにより異なるので各実験に明記。

---

## 1. ベース（34 特徴）

| 項目 | 内容 |
|------|------|
| **構成** | 34 特徴。カテゴリ 8 列（critic_name, production_company, directors, authors, rotten_tomatoes_link, movie_title, movie_info, actors）＋ 数値列（runtime, movie_age_days, release_year/month/dayofweek, review_year/month/dayofweek, top_critic）＋ ジャンル One-hot（genre_Drama 等 8 列）＋ publisher_name, content_rating（category）＋ 3C3 特徴（critic_name_te_ts, production_company_te_ts, 各 TE の 3 ビン化, runtime_bin, movie_age_bin, release_decade）。 |
| **カテゴリ列の扱い** | `fillna("missing").astype("category")` で LightGBM に渡す。LightGBM は「カテゴリ ID」を木の分割で最適にグルーピングして学習。 |
| **数値列の扱い** | そのまま。runtime の欠損は中央値で埋め。 |
| **CV_AUC** | **0.7600**（train_importance / train_text / train_text_vector）。3C3 なしベースは **0.7369**（train_preprocess）／ **0.7356**（train_extended）。 |

---

## 2. 成功した実験（CV 改善）

### 2.1 ts_te — 時系列 TE 追加（critic_name, production_company）

| 項目 | 内容 |
|------|------|
| **対象カラム** | critic_name, production_company |
| **やったこと** | category は**残したまま**、時系列 Target Encoding を**追加**。各行について「その行より前」の同カテゴリの Fresh 率（スムージング m=10）を新列として足す。 |
| **元の値** | critic_name = `"John Doe"`（category ID）。 |
| **処理後の値** | 元の列はそのまま。**新列** `critic_name_te_ts` = **0.72**（0〜1 の実数）。`production_company_te_ts` = **0.58** など。 |
| **CV_AUC** | **0.7425** |
| **差分（ベース比）** | **+0.0056**（ベース 0.7369） |
| **出典** | train_preprocess_experiments.ipynb |

### 2.2 ts_te_binned — 時系列 TE ＋ 3 ビン化追加

| 項目 | 内容 |
|------|------|
| **対象カラム** | critic_name, production_company |
| **やったこと** | ts_te と同じ時系列 TE を追加し、さらにその TE 値を 3 ビン（[0, 0.33)→0, [0.33, 0.67)→1, [0.67, 1.01]→2）に切った列も追加。category は残す。 |
| **元の値** | critic_name = `"John Doe"`（category ID）。 |
| **処理後の値** | 元の列そのまま ＋ `critic_name_te_ts` = 0.72 ＋ **新列** `critic_name_te_ts_bin` = **2**。production_company も同様。 |
| **CV_AUC** | **0.7426** |
| **差分（ベース比）** | **+0.0057**（ベース 0.7369） |
| **出典** | train_preprocess_experiments.ipynb |

### 2.3 ts_te_bins — 時系列 TE ＋ 数値ビン追加

| 項目 | 内容 |
|------|------|
| **対象カラム** | critic_name, production_company, runtime, movie_age_days, release_year |
| **やったこと** | ts_te（critic_name, production_company の時系列 TE 追加）に加え、runtime_bin, movie_age_bin, release_decade を追加。 |
| **元の値** | runtime = 125（数値）、movie_age_days = 400、release_year = 2012。 |
| **処理後の値** | 元の列そのまま ＋ ts_te 分 ＋ **新列** `runtime_bin` = **2**, `movie_age_bin` = **1**, `release_decade` = **2010**。 |
| **CV_AUC** | **0.7425** |
| **差分（ベース比）** | **+0.0056**（ベース 0.7369） |
| **出典** | train_preprocess_experiments.ipynb |

---

## 3. 微増した実験

### 3.1 movie_info → 長さ・語数のみ置き換え

| 項目 | 内容 |
|------|------|
| **対象カラム** | movie_info |
| **やったこと** | movie_info の category を**外し**、文字数と語数の 2 列に**置き換え**。テキスト中身は失われる。 |
| **元の値** | movie_info = `"Always trouble-prone, the life of..."`（category ID）。 |
| **処理後の値** | 元の列は**削除**。**新列** `movie_info_len` = **35**（文字数）, `movie_info_word_count` = **6**（語数）のみ。 |
| **CV_AUC** | **0.7369** |
| **差分（ベース比）** | **+0.0012**（ベース 0.7356） |
| **出典** | train_extended.ipynb |

### 3.2 add_freq_publisher_name — 出版社名の頻度追加

| 項目 | 内容 |
|------|------|
| **対象カラム** | publisher_name |
| **やったこと** | category は**残したまま**、学習データ内の出現回数を**追加**。target は使わない。 |
| **元の値** | publisher_name = `"New York Times"`（category ID）。 |
| **処理後の値** | 元の列そのまま ＋ **新列** `publisher_name_freq` = **350**（整数）。 |
| **CV_AUC** | **0.7603** |
| **差分（ベース比）** | **+0.0003**（ベース 0.7600） |
| **出典** | train_importance_experiments.ipynb |

### 3.3 movie_info_meta — あらすじの長さ・語数を追加

| 項目 | 内容 |
|------|------|
| **対象カラム** | movie_info |
| **やったこと** | movie_info の category は**残したまま**、文字数と語数を**追加**。§3.1 の「置き換え」とは異なり元のテキストは残る。 |
| **元の値** | movie_info = `"Always trouble-prone..."`（category ID）。 |
| **処理後の値** | 元の列そのまま ＋ **新列** `movie_info_len` = **35**, `movie_info_word_count` = **6**。 |
| **CV_AUC** | **0.7602** |
| **差分（ベース比）** | **+0.0003**（ベース 0.7600） |
| **出典** | train_importance_experiments.ipynb |

### 3.4 movie_info_and_title_meta — あらすじ＋タイトルの長さ・語数を追加

| 項目 | 内容 |
|------|------|
| **対象カラム** | movie_info, movie_title |
| **やったこと** | movie_info_meta（§3.3）に加え、movie_title の文字数・語数も**追加**。category は両方残す。 |
| **元の値** | movie_info = `"Always trouble-prone..."`、movie_title = `"Percy Jackson & the Olympians"`。 |
| **処理後の値** | 元の列そのまま ＋ **新列** `movie_info_len`, `movie_info_word_count`, `movie_title_len` = **28**, `movie_title_word_count` = **4**（計 +4 列）。 |
| **CV_AUC** | **0.76022** |
| **差分（ベース比）** | **+0.00026**（ベース 0.75996） |
| **出典** | train_text_experiments.ipynb。**採用・提出済み**。パブリックで約 +0.004 の改善。 |

### 3.5 movie_info_tfidf_svd20 — TF-IDF → SVD 20 次元を追加

| 項目 | 内容 |
|------|------|
| **対象カラム** | movie_info |
| **やったこと** | movie_info を TfidfVectorizer(max_features=200) でベクトル化し、TruncatedSVD(20) で圧縮した 20 次元を**追加**。各 fold の tr のみで fit。 |
| **元の値** | movie_info = `"Always trouble-prone..."`（category ID）。 |
| **処理後の値** | 元の列そのまま ＋ **新列** `svd_mi_0` 〜 `svd_mi_19`（各成分は実数、正負あり）。テキストの粗いトピックに相当。 |
| **CV_AUC** | **0.76012** |
| **差分（ベース比）** | **+0.00016**（ベース 0.75996） |
| **出典** | train_text_experiments.ipynb |

---

## 4. 変化なしの実験

### 4.1 critic_name ＋ 頻度追加

| 項目 | 内容 |
|------|------|
| **対象カラム** | critic_name |
| **やったこと** | category は残したまま、出現回数を**追加**。 |
| **元の値** | critic_name = `"John Doe"`（category ID）。 |
| **処理後の値** | 元の列そのまま ＋ **新列** `critic_name_freq` = **50**（整数）。 |
| **CV_AUC** | **0.7356** |
| **差分（ベース比）** | **0.0000**（ベース 0.7356） |
| **出典** | train_extended.ipynb |

### 4.2 bins — 数値ビンのみ追加（runtime_bin, movie_age_bin, release_decade）

| 項目 | 内容 |
|------|------|
| **対象カラム** | runtime, movie_age_days, release_year |
| **やったこと** | 数値をビンに切った列を**追加**のみ。時系列 TE は足さない。 |
| **元の値** | runtime = 125, movie_age_days = 400, release_year = 2012。 |
| **処理後の値** | 元の列そのまま ＋ **新列** `runtime_bin` = 2, `movie_age_bin` = 1, `release_decade` = 2010。 |
| **CV_AUC** | **0.7369** |
| **差分（ベース比）** | **0.0000**（ベース 0.7369）。※ts_te と組み合わせた ts_te_bins では +0.0056。 |
| **出典** | train_preprocess_experiments.ipynb |

### 4.3 missing_flags — 欠損フラグ追加

| 項目 | 内容 |
|------|------|
| **対象カラム** | runtime, movie_age_days |
| **やったこと** | 欠損なら 1, それ以外は 0 の 2 値列を**追加**。 |
| **元の値** | runtime = NaN or 120。 |
| **処理後の値** | 元の列そのまま ＋ **新列** `is_runtime_missing` = 1 or 0, `is_movie_age_days_missing` = 1 or 0。 |
| **CV_AUC** | **0.7369**（preprocess）／ **0.7600**（importance） |
| **差分（ベース比）** | **0.0000** |
| **出典** | train_preprocess_experiments.ipynb, train_importance_experiments.ipynb |

### 4.4 bins_and_flags — ビン ＋ 欠損フラグ

| 項目 | 内容 |
|------|------|
| **対象カラム** | runtime, movie_age_days, release_year |
| **やったこと** | §4.2 bins ＋ §4.3 missing_flags の組み合わせ。 |
| **元の値** | 同上。 |
| **処理後の値** | runtime_bin, movie_age_bin, release_decade, is_runtime_missing, is_movie_age_days_missing（計 5 列追加）。 |
| **CV_AUC** | **0.7369** |
| **差分（ベース比）** | **0.0000**（ベース 0.7369） |
| **出典** | train_preprocess_experiments.ipynb |

### 4.5 add_freq_multi — 複数列に頻度を一括追加

| 項目 | 内容 |
|------|------|
| **対象カラム** | 複数のカテゴリ列 |
| **やったこと** | 複数列の出現回数を一括で**追加**。 |
| **元の値** | 各カテゴリ列の category ID。 |
| **処理後の値** | 各列に `{col}_freq`（整数）を追加。 |
| **CV_AUC** | **0.7600** |
| **差分（ベース比）** | **0.0000**（ベース 0.7600） |
| **出典** | train_importance_experiments.ipynb |

### 4.6 movie_title_len — タイトル文字数追加

| 項目 | 内容 |
|------|------|
| **対象カラム** | movie_title |
| **やったこと** | movie_title の文字数を 1 列**追加**。category は残す。 |
| **元の値** | movie_title = `"Percy Jackson & the Olympians"`（category ID）。 |
| **処理後の値** | 元の列そのまま ＋ **新列** `movie_title_len` = **28**（整数）。 |
| **CV_AUC** | **0.75996** |
| **差分（ベース比）** | **0.00000**（ベース 0.75996） |
| **出典** | train_text_experiments.ipynb |

### 4.7 movie_title_word_count — タイトル語数追加

| 項目 | 内容 |
|------|------|
| **対象カラム** | movie_title |
| **やったこと** | movie_title の語数を 1 列**追加**。category は残す。 |
| **元の値** | movie_title = `"Percy Jackson & the Olympians"`（category ID）。 |
| **処理後の値** | 元の列そのまま ＋ **新列** `movie_title_word_count` = **4**（整数）。 |
| **CV_AUC** | **0.75996** |
| **差分（ベース比）** | **0.00000**（ベース 0.75996） |
| **出典** | train_text_experiments.ipynb |

---

## 5. 失敗した実験（CV 悪化）

### 5.1 8 列を TE で一括置き換え

| 項目 | 内容 |
|------|------|
| **対象カラム** | critic_name, production_company, directors, authors, rotten_tomatoes_link, movie_title, movie_info, actors の 8 列 |
| **やったこと** | 8 列すべてを「学習データ全体でのそのカテゴリの Fresh 率」に**置き換え**。元の category は**削除**。 |
| **元の値** | 各列 = category ID。 |
| **処理後の値** | 元の列は**削除**。**置き換え列** `{col}_te` = 0〜1 の実数（例: directors_te = 0.72）。 |
| **CV_AUC** | **0.7030** |
| **差分（ベース比）** | **-0.034**（ベース 0.7369） |
| **原因** | Fresh 率という単一軸に依存しすぎ＋時系列のずれで汎化せず。元の category を捨てたため木の柔軟なグルーピングが効かなくなった。 |
| **出典** | train_preprocess_compare.ipynb |

### 5.2 8 列を頻度で一括置き換え

| 項目 | 内容 |
|------|------|
| **対象カラム** | 8 列（§5.1 と同じ） |
| **やったこと** | 8 列すべてを出現回数（整数）に**置き換え**。元の category は**削除**。target は使わない。 |
| **元の値** | 各列 = category ID。 |
| **処理後の値** | 元の列は**削除**。**置き換え列** `{col}_freq` = 整数（例: directors_freq = 42）。 |
| **CV_AUC** | **0.6920** |
| **差分（ベース比）** | **-0.045**（ベース 0.7369） |
| **原因** | target と無関係な出現回数だけでは情報が弱すぎ。8 列すべてを置き換えたため category のシグナルを丸ごと捨てた。 |
| **出典** | train_preprocess_compare.ipynb |

### 5.3 loo — critic_name に LOO 追加

| 項目 | 内容 |
|------|------|
| **対象カラム** | critic_name |
| **やったこと** | category は残したまま、Leave-One-Out（自分を除いた同カテゴリの target 平均、スムージング m=5）を**追加**。 |
| **元の値** | critic_name = `"John Doe"`（category ID）。 |
| **処理後の値** | 元の列そのまま ＋ **新列** `critic_name_loo` = **0.68**（0〜1 の実数）。 |
| **CV_AUC** | **0.7203** |
| **差分（ベース比）** | **-0.0166**（ベース 0.7369） |
| **原因** | 同じ批評家の他レビューの target を使うため実質リークに近い。スムージングしてもノイズ。 |
| **出典** | train_preprocess_experiments.ipynb |

### 5.4 loo_directors — directors に LOO 追加

| 項目 | 内容 |
|------|------|
| **対象カラム** | directors |
| **やったこと** | category は残したまま、LOO（自分を除いた同監督の target 平均）を**追加**。 |
| **元の値** | directors = `"Christopher Nolan"`（category ID）。 |
| **処理後の値** | 元の列そのまま ＋ **新列** `directors_loo` = 0〜1 の実数。 |
| **CV_AUC** | **0.7317** |
| **差分（ベース比）** | **-0.0282**（ベース 0.7600） |
| **原因** | §5.3 と同様、リークに近くノイズ。 |
| **出典** | train_importance_experiments.ipynb |

### 5.5 loo_authors — authors に LOO 追加

| 項目 | 内容 |
|------|------|
| **対象カラム** | authors |
| **やったこと** | category は残したまま、LOO を**追加**。 |
| **元の値** | authors = category ID。 |
| **処理後の値** | 元の列そのまま ＋ **新列** `authors_loo` = 0〜1 の実数。 |
| **CV_AUC** | **0.7062** |
| **差分（ベース比）** | **-0.0538**（ベース 0.7600） |
| **原因** | LOO の中で最悪。authors はカーディナリティが高くスムージングしても不安定。 |
| **出典** | train_importance_experiments.ipynb |

### 5.6 ts_te_directors — directors に時系列 TE 追加

| 項目 | 内容 |
|------|------|
| **対象カラム** | directors |
| **やったこと** | category は残したまま、「その行より前」の同監督の Fresh 率（スムージング m=10）を**追加**。 |
| **元の値** | directors = `"Christopher Nolan"`（category ID）。 |
| **処理後の値** | 元の列そのまま ＋ **新列** `directors_te_ts` = 0〜1 の実数。 |
| **CV_AUC** | **0.7471** |
| **差分（ベース比）** | **-0.0129**（ベース 0.7600） |
| **原因** | critic_name / production_company では効いた時系列 TE が、directors では効かなかった。監督は作品ごとに評価が変わり、過去の Fresh 率が将来を予測しない。 |
| **出典** | train_importance_experiments.ipynb |

### 5.7 ts_te_binned_directors — directors に時系列 TE ＋ 3 ビン化追加

| 項目 | 内容 |
|------|------|
| **対象カラム** | directors |
| **やったこと** | §5.6 の時系列 TE に加え、その値を 3 ビンに切った列も**追加**。 |
| **元の値** | directors = category ID。 |
| **処理後の値** | 元の列そのまま ＋ `directors_te_ts` ＋ **新列** `directors_te_ts_bin` = 0/1/2。 |
| **CV_AUC** | **0.7470** |
| **差分（ベース比）** | **-0.0130**（ベース 0.7600） |
| **原因** | §5.6 と同じ。ビン化しても改善せず。 |
| **出典** | train_importance_experiments.ipynb |

### 5.8 ts_te_authors — authors に時系列 TE 追加

| 項目 | 内容 |
|------|------|
| **対象カラム** | authors |
| **やったこと** | category は残したまま、時系列 TE を**追加**。 |
| **元の値** | authors = category ID。 |
| **処理後の値** | 元の列そのまま ＋ **新列** `authors_te_ts` = 0〜1 の実数。 |
| **CV_AUC** | **0.7385** |
| **差分（ベース比）** | **-0.0215**（ベース 0.7600） |
| **原因** | authors は directors 以上にカーディナリティが高く時系列 TE が不安定。 |
| **出典** | train_importance_experiments.ipynb |

### 5.9 ts_te_binned_authors — authors に時系列 TE ＋ 3 ビン化追加

| 項目 | 内容 |
|------|------|
| **対象カラム** | authors |
| **やったこと** | §5.8 の時系列 TE に加え、3 ビン化列も**追加**。 |
| **元の値** | authors = category ID。 |
| **処理後の値** | 元の列そのまま ＋ `authors_te_ts` ＋ `authors_te_ts_bin`。 |
| **CV_AUC** | **0.7385** |
| **差分（ベース比）** | **-0.0215**（ベース 0.7600） |
| **原因** | §5.8 と同じ。 |
| **出典** | train_importance_experiments.ipynb |

### 5.10 ts_te_multi — 複数列に時系列 TE を一括追加

| 項目 | 内容 |
|------|------|
| **対象カラム** | 複数カテゴリ列 |
| **やったこと** | 複数のカテゴリ列に時系列 TE を一括で**追加**。 |
| **元の値** | 各列 = category ID。 |
| **処理後の値** | 各列に `{col}_te_ts`（0〜1）を追加。 |
| **CV_AUC** | **0.7429** |
| **差分（ベース比）** | **-0.0171**（ベース 0.7600） |
| **原因** | directors / authors など効かない列を含めると全体で悪化。 |
| **出典** | train_importance_experiments.ipynb |

### 5.11 movie_review_count_ts — 映画ごと時系列特徴追加

| 項目 | 内容 |
|------|------|
| **対象カラム** | rotten_tomatoes_link（映画 ID） |
| **やったこと** | category は残したまま、「その行より前」の同映画のレビュー数と過去 Fresh 率を**追加**。 |
| **元の値** | rotten_tomatoes_link = `"m/0814255"`（category ID）。 |
| **処理後の値** | 元の列そのまま ＋ **新列** `movie_review_count_ts` = **12**（整数）, `movie_fresh_rate_ts` = **0.65**（実数）。 |
| **CV_AUC** | **0.7423** |
| **差分（ベース比）** | **-0.0177**（ベース 0.7600） |
| **原因** | 映画ごとの時系列集計がノイズになった。 |
| **出典** | train_importance_experiments.ipynb |

### 5.12 directors ＋ 頻度追加

| 項目 | 内容 |
|------|------|
| **対象カラム** | directors |
| **やったこと** | category は残したまま、出現回数を**追加**。 |
| **元の値** | directors = `"Christopher Nolan"`（category ID）。 |
| **処理後の値** | 元の列そのまま ＋ **新列** `directors_freq` = **42**（整数）。 |
| **CV_AUC** | **0.7277** |
| **差分（ベース比）** | **-0.0080**（ベース 0.7356） |
| **原因** | 出現回数が Fresh/Rotten と無関係でノイズ。 |
| **出典** | train_extended.ipynb |

### 5.13 authors ＋ 頻度追加

| 項目 | 内容 |
|------|------|
| **対象カラム** | authors |
| **やったこと** | category は残したまま、出現回数を**追加**。 |
| **元の値** | authors = category ID。 |
| **処理後の値** | 元の列そのまま ＋ **新列** `authors_freq` = 整数。 |
| **CV_AUC** | **0.7291** |
| **差分（ベース比）** | **-0.0065**（ベース 0.7356） |
| **原因** | §5.12 と同様。 |
| **出典** | train_extended.ipynb |

### 5.14 add_freq_rotten_tomatoes_link — 映画 ID の頻度追加

| 項目 | 内容 |
|------|------|
| **対象カラム** | rotten_tomatoes_link |
| **やったこと** | category は残したまま、出現回数を**追加**。 |
| **元の値** | rotten_tomatoes_link = `"m/0814255"`（category ID）。 |
| **処理後の値** | 元の列そのまま ＋ **新列** `rotten_tomatoes_link_freq` = **100**（整数）。 |
| **CV_AUC** | **0.7546** |
| **差分（ベース比）** | **-0.0054**（ベース 0.7600） |
| **原因** | レビュー数はすでに他の特徴量で捉えられており、追加してもノイズ。 |
| **出典** | train_importance_experiments.ipynb |

### 5.15 add_freq_movie_title — タイトル頻度追加

| 項目 | 内容 |
|------|------|
| **対象カラム** | movie_title |
| **やったこと** | category は残したまま、出現回数を**追加**。 |
| **元の値** | movie_title = `"Percy Jackson..."`（category ID）。 |
| **処理後の値** | 元の列そのまま ＋ **新列** `movie_title_freq` = **3**（整数）。 |
| **CV_AUC** | **0.7551** |
| **差分（ベース比）** | **-0.0049**（ベース 0.7600） |
| **原因** | ほぼユニークなので頻度は 1〜数回ばかりで情報量がない。 |
| **出典** | train_importance_experiments.ipynb |

### 5.16 add_freq_actors — 出演者頻度追加

| 項目 | 内容 |
|------|------|
| **対象カラム** | actors |
| **やったこと** | category は残したまま、出現回数を**追加**。 |
| **元の値** | actors = category ID。 |
| **処理後の値** | 元の列そのまま ＋ **新列** `actors_freq` = 整数。 |
| **CV_AUC** | **0.7562** |
| **差分（ベース比）** | **-0.0038**（ベース 0.7600） |
| **原因** | 出演回数が Fresh/Rotten と無関係。 |
| **出典** | train_importance_experiments.ipynb |

### 5.17 add_freq_production_company — 制作会社頻度追加

| 項目 | 内容 |
|------|------|
| **対象カラム** | production_company |
| **やったこと** | category は残したまま、出現回数を**追加**。 |
| **元の値** | production_company = category ID。 |
| **処理後の値** | 元の列そのまま ＋ **新列** `production_company_freq` = 整数。 |
| **CV_AUC** | **0.7598** |
| **差分（ベース比）** | **-0.0001**（ベース 0.7600） |
| **原因** | ほぼ変化なし。制作会社の出現回数はシグナルにならなかった。 |
| **出典** | train_importance_experiments.ipynb |

### 5.18 movie_title → 頻度のみに置き換え

| 項目 | 内容 |
|------|------|
| **対象カラム** | movie_title |
| **やったこと** | movie_title の category を**外し**、出現回数のみに**置き換え**。タイトル文字列は失われる。 |
| **元の値** | movie_title = `"Percy Jackson..."`（category ID）。 |
| **処理後の値** | 元の列は**削除**。**置き換え列** `movie_title_freq` = **3**（整数）のみ。 |
| **CV_AUC** | **0.7339**（train_extended）／ **0.7326**（train_preprocess） |
| **差分（ベース比）** | **-0.0018**（ベース 0.7356）／ **-0.0043**（ベース 0.7369） |
| **原因** | タイトル情報を捨て、出現回数（ほぼ 1〜数回）だけにしてシグナルを失った。 |
| **出典** | train_extended.ipynb, train_preprocess_experiments.ipynb |

### 5.19 movie_info_meta 置き換え — あらすじを長さ・語数のみに置き換え（別実験）

| 項目 | 内容 |
|------|------|
| **対象カラム** | movie_info |
| **やったこと** | movie_info の category を**外し**、文字数と語数のみに**置き換え**。§3.1 とは別ノートで実施。 |
| **元の値** | movie_info = `"Always trouble-prone..."`（category ID）。 |
| **処理後の値** | 元の列は**削除**。**新列** `movie_info_len`, `movie_info_word_count` のみ。 |
| **CV_AUC** | **0.7365** |
| **差分（ベース比）** | **-0.0004**（ベース 0.7369） |
| **原因** | §3.1（train_extended）では微増だったが、こちらのノートでは微減。環境差。置き換えで category の情報を捨てたことが原因。 |
| **出典** | train_preprocess_experiments.ipynb |

### 5.20 all_meta_freq — メタ＋頻度＋ビンの組み合わせ

| 項目 | 内容 |
|------|------|
| **対象カラム** | movie_info, movie_title, runtime, movie_age_days, release_year |
| **やったこと** | movie_info → 長さ・語数のみに**置き換え** ＋ movie_title → 頻度のみに**置き換え** ＋ runtime_bin, movie_age_bin, release_decade を追加。 |
| **元の値** | movie_info, movie_title は category ID。runtime 等は数値。 |
| **処理後の値** | movie_info, movie_title は**削除**。代わりに movie_info_len, movie_info_word_count, movie_title_freq, runtime_bin, movie_age_bin, release_decade。 |
| **CV_AUC** | **0.7350** |
| **差分（ベース比）** | **-0.0019**（ベース 0.7369） |
| **原因** | 置き換えによる情報削減の組み合わせ。失敗した置き換えを重ねてノイズが増えた。 |
| **出典** | train_preprocess_experiments.ipynb |

---

## 6. テキストベクトル化の網羅実験（途中）

train_text_vector_experiments.ipynb / text_vectors.py。ベース 34 特徴。時系列 CV。各 fold の tr のみでベクトル化を fit。ベース CV_AUC = **0.7600**。**nmf_20 の途中で停止**。詳細は docs/TRAIN_TEXT_VECTOR_EXPERIMENTS_RESULTS.md。

### 6.1 tfidf_50 — TF-IDF 50 次元追加

| 項目 | 内容 |
|------|------|
| **対象カラム** | movie_info |
| **やったこと** | movie_info を TfidfVectorizer(max_features=50) でベクトル化し、50 列を**追加**。 |
| **元の値** | movie_info = テキスト（category ID）。 |
| **処理後の値** | 元の列そのまま ＋ **新列** `tfidf_0` 〜 `tfidf_49`（0〜1 に近い実数）。 |
| **CV_AUC** | **0.7600** |
| **差分（ベース比）** | **0.0000** |
| **出典** | train_text_vector_experiments.ipynb |

### 6.2 tfidf_100 — TF-IDF 100 次元追加

| 項目 | 内容 |
|------|------|
| **対象カラム** | movie_info |
| **やったこと** | TfidfVectorizer(max_features=100) で 100 列を**追加**。 |
| **元の値** | movie_info = テキスト。 |
| **処理後の値** | 元の列そのまま ＋ `tfidf_0` 〜 `tfidf_99`。 |
| **CV_AUC** | **0.7600** |
| **差分（ベース比）** | **0.0000** |
| **出典** | train_text_vector_experiments.ipynb |

### 6.3 tfidf_svd20 — TF-IDF → SVD 20 次元追加

| 項目 | 内容 |
|------|------|
| **対象カラム** | movie_info |
| **やったこと** | TF-IDF(200) → TruncatedSVD(20) で 20 列を**追加**。 |
| **元の値** | movie_info = テキスト。 |
| **処理後の値** | 元の列そのまま ＋ `tfidf_svd_0` 〜 `tfidf_svd_19`（実数、正負あり）。 |
| **CV_AUC** | **0.7601** |
| **差分（ベース比）** | **+0.0001** |
| **出典** | train_text_vector_experiments.ipynb |

### 6.4 count_50 — CountVectorizer 50 次元追加

| 項目 | 内容 |
|------|------|
| **対象カラム** | movie_info |
| **やったこと** | CountVectorizer(max_features=50) で出現回数ベースの 50 列を**追加**。IDF による重み付けなし（BoW）。 |
| **元の値** | movie_info = テキスト。 |
| **処理後の値** | 元の列そのまま ＋ `count_0` 〜 `count_49`（0 以上の整数）。 |
| **CV_AUC** | **0.7603** |
| **差分（ベース比）** | **+0.0003** |
| **出典** | train_text_vector_experiments.ipynb |

### 6.5 count_svd20 — Count → SVD 20 次元追加

| 項目 | 内容 |
|------|------|
| **対象カラム** | movie_info |
| **やったこと** | Count(200) → TruncatedSVD(20) で 20 列を**追加**。 |
| **元の値** | movie_info = テキスト。 |
| **処理後の値** | 元の列そのまま ＋ `count_svd_0` 〜 `count_svd_19`。 |
| **CV_AUC** | **0.7603** |
| **差分（ベース比）** | **+0.0003** |
| **出典** | train_text_vector_experiments.ipynb |

### 6.6 hash_50 — HashingVectorizer 50 次元追加

| 項目 | 内容 |
|------|------|
| **対象カラム** | movie_info |
| **やったこと** | HashingVectorizer(n_features=512) の先頭 50 列を**追加**。語彙を持たないため省メモリ。 |
| **元の値** | movie_info = テキスト。 |
| **処理後の値** | 元の列そのまま ＋ `hash_0` 〜 `hash_49`（実数）。 |
| **CV_AUC** | **0.7600** |
| **差分（ベース比）** | **0.0000** |
| **出典** | train_text_vector_experiments.ipynb |

### 6.7 hash_svd20 — Hash → SVD 20 次元追加

| 項目 | 内容 |
|------|------|
| **対象カラム** | movie_info |
| **やったこと** | HashingVectorizer(512) → TruncatedSVD(20) で 20 列を**追加**。 |
| **元の値** | movie_info = テキスト。 |
| **処理後の値** | 元の列そのまま ＋ hash_svd 系 20 列。 |
| **CV_AUC** | **0.7601** |
| **差分（ベース比）** | **+0.0001** |
| **出典** | train_text_vector_experiments.ipynb |

### 6.8 lda_10 — LDA 10 トピック追加

| 項目 | 内容 |
|------|------|
| **対象カラム** | movie_info |
| **やったこと** | TF-IDF 上で LDA(n_components=10) を fit し、10 列を**追加**。 |
| **元の値** | movie_info = テキスト。 |
| **処理後の値** | 元の列そのまま ＋ lda_0 〜 lda_9（トピック混合比）。 |
| **CV_AUC** | **0.7606** |
| **差分（ベース比）** | **+0.0006** |
| **出典** | train_text_vector_experiments.ipynb |

### 6.9 lda_20 — LDA 20 トピック追加

| 項目 | 内容 |
|------|------|
| **対象カラム** | movie_info |
| **やったこと** | LDA(n_components=20) で 20 列を**追加**。 |
| **元の値** | movie_info = テキスト。 |
| **処理後の値** | 元の列そのまま ＋ lda_0 〜 lda_19。 |
| **CV_AUC** | **0.7603** |
| **差分（ベース比）** | **+0.0003** |
| **出典** | train_text_vector_experiments.ipynb |

### 6.10 nmf_10 — NMF 10 次元追加

| 項目 | 内容 |
|------|------|
| **対象カラム** | movie_info |
| **やったこと** | TF-IDF 上で NMF(n_components=10) を fit し、10 列を**追加**。 |
| **元の値** | movie_info = テキスト。 |
| **処理後の値** | 元の列そのまま ＋ nmf_0 〜 nmf_9。 |
| **CV_AUC** | **0.7600** |
| **差分（ベース比）** | **0.0000** |
| **出典** | train_text_vector_experiments.ipynb |

### 6.11 nmf_20 以降 — 要再開

**nmf_20** は実行途中で停止（未記録）。続きは **nmf_20 から** 再開する。  
未実施: nmf_20, doc2vec_32, word2vec_32, sentence_transformer_32, concat_mi_title_tfidf_50, concat_mi_title_tfidf_svd20, concat_mi_title_count_50。ノートの「再開用」セルで `RECORDED_PARTIAL_RESULTS` と `START_FROM_CONFIG = "nmf_20"` を設定して CV セルを実行。詳細は docs/TRAIN_TEXT_VECTOR_EXPERIMENTS_RESULTS.md。

---

## 7. 技術別アルゴリズムと値の変化

### 7.1 時系列 Target Encoding（ts_te）

| 項目 | 内容 |
|------|------|
| **何をするか** | **その行より前のデータだけ**で、カテゴリ別の Fresh 率を計算し、スムージングして 1 個の実数にする。**既存列は残し、新列を追加**。 |
| **式（tr 内）** | `review_date` で昇順ソート。行 i について: past_sum = 同カテゴリの「自分より前」の target 合計、past_count = 同カテゴリの「自分より前」の件数。te_i = (past_sum + m × 全体平均) / (past_count + m)。m=10。past_count が 0 なら全体平均。 |
| **式（val/test）** | tr 全体でカテゴリ別のスムージング平均を計算し、その行のカテゴリに対応する値を割り当て。未知カテゴリは全体平均。 |
| **元の値 → 変換後の値** | `critic_name = "John Doe"` → **新列** `critic_name_te_ts = 0.72`（0〜1）。 |

### 7.2 時系列 TE の 3 ビン化（ts_te_binned）

| 項目 | 内容 |
|------|------|
| **何をするか** | 時系列 TE の値を 3 つのビンに切った**追加列**。単一軸依存を弱める。 |
| **式** | [0, 0.33) → 0、[0.33, 0.67) → 1、[0.67, 1.01] → 2。欠損は 1。 |
| **元の値 → 変換後の値** | `critic_name_te_ts = 0.72` → `critic_name_te_ts_bin = 2`。 |

### 7.3 頻度エンコード（freq）

| 項目 | 内容 |
|------|------|
| **何をするか** | target は使わず、**学習データ内でそのカテゴリが何回出現したか**を 1 個の整数にする。 |
| **式** | tr で `value_counts()`。各行に出現回数を割り当て。val/test は tr の辞書でマッピング。未知は 0。 |
| **元の値 → 変換後の値** | `directors = "Christopher Nolan"` → `directors_freq = 42`。 |

### 7.4 Leave-One-Out（LOO）

| 項目 | 内容 |
|------|------|
| **何をするか** | 各行について、**その行を除いた**同カテゴリの target 平均でエンコード。スムージング（m=5）あり。 |
| **式** | loo_i = (同カテゴリの target 合計 − 自分) / (同カテゴリ件数 − 1)。1 件なら全体平均。smooth = (count·loo + m·全体平均)/(count + m)。 |
| **元の値 → 変換後の値** | `critic_name = "John Doe"` → `critic_name_loo = 0.68`。 |

### 7.5 ビニング

| 項目 | 内容 |
|------|------|
| **何をするか** | 数値を区間に切って離散ラベル（0,1,2,...）にする**追加列**。 |
| **式** | runtime_bin: [0,90)→0, [90,120)→1, [120,150)→2, [150,1000)→3。欠損は 1。movie_age_bin: 0→0, <365→1, <1825→2, <7300→3, 以上→4。release_decade: (release_year // 10) * 10。 |
| **元の値 → 変換後の値** | runtime=125 → runtime_bin=2。movie_age_days=400 → movie_age_bin=1。release_year=2012 → release_decade=2010。 |

### 7.6 欠損フラグ

| 項目 | 内容 |
|------|------|
| **何をするか** | 欠損なら 1、それ以外は 0 の**追加列**。 |
| **式** | is_runtime_missing: NaN なら 1 else 0。is_movie_age_days_missing: 同様。 |
| **元の値 → 変換後の値** | runtime=NaN → is_runtime_missing=1。runtime=120 → 0。 |

### 7.7 テキストの長さ・語数（movie_info_meta / movie_title_len 等）

| 項目 | 内容 |
|------|------|
| **何をするか** | テキスト列の文字数・語数を**追加列**にする。 |
| **式** | len = str(text).len()。word_count = str(text).split() の長さ。欠損は 0。 |
| **元の値 → 変換後の値** | movie_info="Always trouble-prone..." → movie_info_len=35, movie_info_word_count=6。 |

### 7.8 TF-IDF

| 項目 | 内容 |
|------|------|
| **何をするか** | テキストを単語（＋バイグラム）の TF-IDF ベクトルにし、上位 n 次元を**追加列**にする。各 fold の tr のみで fit。 |
| **式** | TfidfVectorizer(max_features=n, min_df=2, max_df=0.95, ngram_range=(1,2), sublinear_tf=True)。 |
| **元の値 → 変換後の値** | テキスト → tfidf_0, tfidf_1, ... がそれぞれ 0〜1 に近い実数。 |

### 7.9 TF-IDF → TruncatedSVD

| 項目 | 内容 |
|------|------|
| **何をするか** | TF-IDF 行列を SVD で低次元に圧縮。 |
| **式** | TruncatedSVD(n_components=20, random_state=42)。 |
| **元の値 → 変換後の値** | テキスト → svd_0 〜 svd_19（実数、正負あり）。テキストの粗いトピック。 |

### 7.10 CountVectorizer

| 項目 | 内容 |
|------|------|
| **何をするか** | テキストを単語の出現回数ベクトルにし、上位 n 次元を**追加列**にする。IDF 重み付けなし（BoW）。 |
| **式** | CountVectorizer(max_features=n, min_df=2, max_df=0.95)。 |
| **元の値 → 変換後の値** | テキスト → count_0, count_1, ...（0 以上の整数）。 |

### 7.11 HashingVectorizer

| 項目 | 内容 |
|------|------|
| **何をするか** | テキストをハッシュ関数で固定次元のスパースベクトルにし、先頭 n 列を使う。語彙を持たず省メモリ。 |
| **式** | HashingVectorizer(n_features=512)。先頭 50 列などを使用。 |
| **元の値 → 変換後の値** | テキスト → hash_0, hash_1, ...（実数）。 |

### 7.12 sentence-transformers

| 項目 | 内容 |
|------|------|
| **何をするか** | 事前学習モデル（例: all-MiniLM-L6-v2）で文を 384 次元ベクトルに encode し、先頭 32 次元を**追加列**にする。 |
| **式** | model.encode(テキストリスト)[:, :32]。 |
| **元の値 → 変換後の値** | テキスト → st_0, st_1, ..., st_31（実数）。意味的に近い文は近いベクトル。 |

### 7.13 Target Encoding（一括置き換え・失敗したやり方）

| 項目 | 内容 |
|------|------|
| **何をするか** | カテゴリを「学習データ全体の Fresh 率」に**置き換え**（元の列を消す）。 |
| **式** | smooth(c) = (n(c)·ȳ(c) + m·全体平均) / (n(c)+m)。m=10。 |
| **元の値 → 変換後の値** | directors="Christopher Nolan" → directors_te=0.72。元の category は消える。 |

---

## 8. 用語・参照

### 8.1 前処理 vs 特徴量追加

| 用語 | 意味 |
|------|------|
| **前処理（置き換え）** | 既存の列を**変換して列を置き換える**。元の列は消える。 |
| **特徴量追加** | 既存の列は**そのまま残し**、新しい列を**足す**。 |

**スコアが伸びた**のはいずれも「特徴量追加」。**落ちた**のは「置き換え」か LOO 追加。

### 8.2 実験ノート・ファイル

| 対象 | ファイル |
|------|----------|
| 前処理比較（時系列 TE・LOO・ビン・頻度など） | train_preprocess_experiments.ipynb |
| 一括置き換え（TE・頻度）の比較 | train_preprocess_compare.ipynb |
| 既存実験（1 種類ずつ前処理を変更） | train_extended.ipynb |
| 重要度ベース「ベース＋1つ」追加 | train_importance_experiments.ipynb |
| テキスト（movie_info / movie_title の TF-IDF・SVD・メタ） | train_text_experiments.ipynb |
| テキストベクトル化の網羅 | train_text_vector_experiments.ipynb, text_vectors.py |
| 共通エンコーディング | experiment_encodings.py |

### 8.3 繰り返さない実験

- 8 列を TE / 頻度で一括置き換え（§5.1, §5.2）
- movie_title を頻度のみに置き換え（§5.18）
- directors / authors / actors の頻度追加（§5.12, §5.13, §5.16）
- LOO 追加（§5.3, §5.4, §5.5）
- directors / authors の時系列 TE（§5.6〜§5.9）
- ts_te_multi（§5.10）

### 8.4 stage5 がパイプラインに反映されていない件

FEATURES.md で定義されている stage_5（release_decade, runtime_bin, movie_age_bin、欠損フラグ、movie_fresh_rate_te, first_director_te など）は、現状の feature_engineering/features.py の `create_features` には実装されておらず、ノート側で FEATURE_STAGES として段階追加するコードも features.py と連携していない。
