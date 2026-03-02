# 特徴量エンジニアリング編 — 方針・実験・結果

前処理で得た「置き換えより足す」「効いた TE だけ追加」を踏まえ、**追加特徴量（掛け合わせ）** を試した結果と、**何の特徴量に何をしてどうなったか** をまとめる。

---

## 1. 方針（ここまでで得た原則）

| 方針 | 内容 |
|------|------|
| **置き換えより「足す」** | 既存列を TE や頻度に置き換えると CV が悪化。効いたのはすべて「既存を残して新列を足す」。 |
| **高カーディナリティはそのまま** | directors, authors, actors は **category のまま** が有効。これらに TE や頻度を足すと悪化したので、変えるなら「足す」に限定（例: 交互作用 1 列）。 |
| **効いた列だけ時系列 TE を追加** | critic_name, production_company の時系列 TE 追加は維持。directors / authors には広げない。 |
| **テキストは微増止まり** | あらすじの TF-IDF / LDA 等は +0.0001〜0.0006 程度。伸ばすならメタ情報の掛け合わせを優先。 |

### やる / やらない一覧

| やる | 時系列 TE の**追加**（critic_name, production_company）。ビン（runtime_bin, movie_age_bin, release_decade）の**追加**。category は残す。 |
| やらない | 8 列を TE や頻度で**一括置き換え**。directors / authors に時系列 TE や LOO を**追加**。movie_title を頻度のみに**置き換え**。critic_name に LOO を追加。 |

---

## 2. 現状のベース（38 列）の成り立ち

モデルに渡す前までに、次の順で列が用意される。

| 段階 | 対象・処理 | 追加される列（例） |
|------|------------|-------------------|
| create_features | 日付分解・ジャンル One-hot・カテゴリ | review_year/month/dayofweek, release_year/month/dayofweek, movie_age_days, genre_*（8列）, content_rating, publisher_name |
| 3C3 | 時系列 TE（critic_name, production_company） | critic_name_te_ts, production_company_te_ts |
| 3C3 | 上記 TE の 3 ビン化 | critic_name_te_ts_bin, production_company_te_ts_bin |
| 3C3 | 数値のビン | runtime_bin, movie_age_bin, release_decade |
| experiment_encodings.movie_info_meta | movie_info の長さ・語数（元列は残す） | movie_info_len, movie_info_word_count |
| ノート内 | movie_title の長さ・語数（元列は残す） | movie_title_len, movie_title_word_count |

**ベース 38 列**: rotten_tomatoes_link, critic_name, top_critic, publisher_name, movie_title, movie_info, content_rating, directors, authors, actors, runtime, production_company, review_year, review_month, review_dayofweek, release_year, release_month, release_dayofweek, movie_age_days, genre_Drama〜Documentary（8列）, critic_name_te_ts, production_company_te_ts, critic_name_te_ts_bin, production_company_te_ts_bin, runtime_bin, movie_age_bin, release_decade, movie_info_len, movie_info_word_count, movie_title_len, movie_title_word_count.

---

## 3. 15パターン追加特徴量 — 対象・処理・追加列・結果

いずれも **既存列は触らず 1 列を追加**。train / test 両方に同じ処理。正規化は **train の統計のみ** を使用（リーク防止）。

### 3.1 一覧表（対象特徴量・やったこと・追加列）

| パターン | 対象となる特徴量 | 行った処理 | 追加した列 |
|----------|------------------|------------|------------|
| P01 | critic_name_te_ts, genre_Drama | そのまま掛け算 | critic_te_x_genre_Drama |
| P02 | critic_name_te_ts, genre_Comedy | そのまま掛け算 | critic_te_x_genre_Comedy |
| P03 | critic_name_te_ts, genre_Documentary | そのまま掛け算 | critic_te_x_genre_Documentary |
| P04 | production_company_te_ts, release_decade | release_decade を train の min〜max で [0,1] 正規化 → release_decade_norm を作成し TE と掛け算 | release_decade_norm（中間）, prod_te_x_release_decade |
| P05 | critic_name_te_ts, runtime_bin | そのまま掛け算 | critic_te_x_runtime_bin |
| P06 | production_company_te_ts, runtime_bin | そのまま掛け算 | prod_te_x_runtime_bin |
| P07 | critic_name_te_ts, movie_age_bin | そのまま掛け算 | critic_te_x_movie_age_bin |
| P08 | production_company_te_ts, genre_Drama | そのまま掛け算 | prod_te_x_genre_Drama |
| P09 | release_decade_norm（P04で作成）, genre_Documentary | そのまま掛け算 | release_decade_x_genre_Documentary |
| P10 | top_critic（bool）, critic_name_te_ts | top_critic を 0/1 に変換してから掛け算 | top_critic_x_critic_te |
| P11 | runtime_bin, genre_Drama | そのまま掛け算 | runtime_bin_x_genre_Drama |
| P12 | movie_age_bin, genre_Documentary | そのまま掛け算 | movie_age_bin_x_genre_Documentary |
| P13 | critic_name_te_ts, movie_title_len | movie_title_len を train の max で [0,1] 正規化 → movie_title_len_norm を作成し TE と掛け算 | movie_title_len_norm（中間）, critic_te_x_title_len |
| P14 | production_company_te_ts, movie_age_bin | そのまま掛け算 | prod_te_x_movie_age_bin |
| P15 | review_year, critic_name_te_ts | review_year を train の min〜max で [0,1] 正規化 → review_year_norm を作成し TE と掛け算 | review_year_norm（中間）, review_year_norm_x_critic_te |

### 3.2 単体実験の結果（ベース + 1パターン）

時系列 CV（val years = 2013〜2016）。ベース AUC = **0.7602**。

| config | n_feat | CV_AUC | std | diff_vs_base |
|--------|--------|--------|-----|--------------|
| p03_critic_te_x_genre_Documentary | 39 | 0.7612 | 0.0049 | **+0.0010** |
| p15_review_year_x_critic_te | 39 | 0.7608 | 0.0050 | **+0.0006** |
| p07_critic_te_x_movie_age_bin | 39 | 0.7606 | 0.0042 | **+0.0004** |
| p11_runtime_bin_x_genre_Drama | 39 | 0.7605 | 0.0059 | **+0.0002** |
| p05_critic_te_x_runtime_bin | 39 | 0.7604 | 0.0042 | **+0.0002** |
| p14_prod_te_x_movie_age_bin | 39 | 0.7604 | 0.0057 | **+0.0002** |
| p01_critic_te_x_genre_Drama | 39 | 0.7604 | 0.0054 | **+0.0002** |
| p04_prod_te_x_release_decade | 39 | 0.7603 | 0.0056 | **+0.0001** |
| p09_release_decade_x_genre_Doc | 39 | 0.7603 | 0.0052 | **+0.0001** |
| base | 38 | 0.7602 | 0.0056 | 0.0000 |
| p02, p10, p12, p13 | 39 | 0.7602 | 0.0056 | 0.0000 |
| p08_prod_te_x_genre_Drama | 39 | 0.7600 | 0.0051 | -0.0003 |
| p06_prod_te_x_runtime_bin | 39 | 0.7594 | 0.0054 | -0.0008 |

**改善した 9 パターン**（diff_vs_base > 0）: p03 → p15 → p07 → p11 → p05 → p14 → p01 → p04 → p09（AUC 順）。

---

## 4. 段階的組み合わせ実験

改善した 9 パターンを「1個ずつ足す」形で段階的に増やし、CV AUC の変化を確認する。

- **やり方**: `train_feature_engineering.ipynb` の「段階的組み合わせ実験」セルを、**15パターン作成＋単体 CV のセル実行後に**実行する。
- **ステージ**: stage0_base（38列）→ stage1（+p03）→ stage2（+p03+p15）→ … → stage9（+9パターン、47列）。
- **追加列の順番**（単体 AUC の良い順）: p03 → p15 → p07 → p11 → p05 → p14 → p01 → p04 → p09。
- 実行後、セル出力の表をこの節に転記して「段階的組み合わせの結果」として残す。

---

## 5. 処理の意図（簡潔）

| パターン | 意図 |
|----------|------|
| P01〜P03, P08 | 批評家・制作会社の「傾向」（TE）がジャンルごとに効き方が違うという仮説。 |
| P04, P09 | 制作会社の傾向と年代、あるいは年代とドキュメンタリーの相性。 |
| P05, P06, P07, P11, P12, P14 | 傾向や上映時間・映画の古さのビンと、他の軸（TE・ジャンル）の交互作用。 |
| P10 | トップ批評家かどうかと、その批評家の傾向の交互作用。 |
| P13 | 批評家の傾向と、タイトル長（正規化）の交互作用。 |
| P15 | レビュー年（正規化）と批評家 TE で、時間トレンドを 1 列にまとめる。 |

---

## 6. 今後の作戦

- 単体では伸びに限界があるため、**掛け合わせ（交互作用）** を少数ずつ足して CV で比較する。
- 候補: 批評家の傾向 × ジャンル、制作会社の傾向 × 年代、runtime_bin × ジャンル など、**メタ情報同士の掛け合わせ**を優先。
- 一気に大量に足すと過学習しやすいので、1〜2 本ずつ試す。

---

## 7. 参照

| 内容 | ファイル |
|------|----------|
| 前処理・テキストベクトル化の実験と結果 | docs/01_PREPROCESS.md |
| ノート本体 | train_feature_engineering.ipynb |
| 特徴量の意味・なぜ使うか | feature_engineering/FEATURES.md |
