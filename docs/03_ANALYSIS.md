# 分析結果 — 現状の改善余地と次に試すこと

01_PREPROCESS.md と 02_FEATURE_ENGINEERING.md を踏まえ、**現状実装でまだ改善できそうなところ**を洗い出し、**次に試すべきこと**を優先度付きでまとめる。

---

## 1. 現状サマリ（01・02 から）

| 項目 | 内容 |
|------|------|
| **ベース** | 38 列（create_features の日付・ジャンル・カテゴリ ＋ ノート内の 3C3 ＋ movie_info_meta ＋ movie_title 長さ・語数）。CV_AUC 約 **0.7602**。 |
| **効いたこと** | 時系列 TE 追加（critic_name, production_company）。ビン追加（runtime_bin, movie_age_bin, release_decade）。既存を残して「足す」だけ。 |
| **効かなかったこと** | 8 列の TE/頻度置き換え、directors/authors の TE・LOO・頻度、movie_title の頻度置き換え、critic_name の LOO。 |
| **テキスト** | movie_info の TF-IDF/LDA 等は +0.0001〜0.0006 の微増。lda_10 が最大 +0.0006。 |
| **掛け合わせ** | 15 パターン中 9 パターンがベースより改善。ベスト単体は p03（critic_te × genre_Documentary）で +0.0010。9 個を段階的に足す実験はノートで実行可能だが、**結果が 02 に未反映**。 |

---

## 2. 実装のまだ改善できそうなところ（洗い出し）

### 2.1 パイプライン・コードまわり

| 観点 | 現状 | 改善の余地 |
|------|------|------------|
| **特徴量定義の分散** | 38 列のうち、create_features は日付・ジャンル・content_rating, publisher_name のみ。3C3・テキストメタ・カテゴリ 8 列の扱いは **train_baseline / train_feature_engineering のノート内** に直書き。feature_engineering/features.py の FEATURES は 19 列で、ノートの 38 列と一致していない。 | 運用ベースを「ノートの FEATURES リスト」に揃えたまま、**どこまでを create_features / experiment_encodings に寄せるか** を整理すると、再現や Kaggle 提出時のミスが減る。必須ではない。 |
| **3C3 の再利用** | 時系列 TE とビンは train_baseline / train_feature_engineering のノートに同じコードが 2 回書かれている。 | experiment_encodings に「3C3 用の時系列 TE ＋ ビン」を関数化して呼び出すと、ノートが短くなり変更漏れを防げる。 |
| **欠損処理** | runtime は train の中央値で埋め。movie_age_days の負値は NaN にしている。 | 01 では欠損フラグは「変化なし」。現状の欠損処理で十分。特に対応不要。 |

### 2.2 まだ取り込んでいない「効いた」追加

| 候補 | 01・02 での結果 | 現状 | 改善の余地 |
|------|-----------------|------|------------|
| **publisher_name_freq** | 01: category は残したまま出現回数を**追加**で +0.0003。 | ベース 38 列には入っていない。 | 1 列足すだけなので、**ノートで追加して CV 比較**すれば取り込み可否を判断できる。 |
| **テキストベクトル lda_10** | 01: movie_info に LDA(10) を**追加**で +0.0006（テキスト系で最大）。 | ベースはテキストメタ（長さ・語数）のみ。TF-IDF/LDA は train_text_vector_experiments で比較したが、ベースライン側には組み込んでいない。 | ベース 38 ＋ lda_10 を **train_baseline や train_feature_engineering の 1 設定として追加**し、CV で 38 のみと比較する価値あり。 |
| **改善 9 パターンの採用** | 02: p03 単体で +0.0010。9 パターンは「段階的組み合わせ」でノート実行可能。 | train_baseline は 38 列のみ。train_feature_engineering で 9 パターン追加した場合の **段階的結果（stage1〜9 の AUC）が 02 に未記載**。 | **段階的組み合わせを 1 回実行し、結果を 02 に転記**。そのうえで「ベースラインを 38＋p03」や「38＋改善 9 列」にするか決めるとよい。 |

### 2.3 実験・検証まわり

| 観点 | 現状 | 改善の余地 |
|------|------|------------|
| **段階的組み合わせの結果** | 02 §4 に「実行後、セル出力の表を転記」とあるが未実施。 | **train_feature_engineering の段階的組み合わせセルを実行し、stage0〜9 の CV_AUC を 02 に追記**する。 |
| **テキストベクトル未実施 config** | 01: nmf_20, concat_mi_title_* , doc2vec_32, word2vec_32, sentence_transformer_32 は未記録または未実施。 | 優先度は低い（テキストは微増止まり）。再開するなら train_text_vector_experiments の「再開用」で nmf_20 から。 |
| **ハイパーパラメータ** | LGB は n_estimators=100, learning_rate=0.1, num_leaves=31, early_stopping=20 で固定。 | チューニングはまだしていない。**特徴量を固めたあと**に、optuna 等で n_estimators / learning_rate / num_leaves などを探索する余地あり。 |
| **検証の切り方** | 時系列 CV（val years = 2013〜2016）のみ。 | 同じ切り方で一貫しているので問題なし。別の val 年や holdout を足す必要は現時点では薄い。 |

### 2.4 データ・リーク・その他

| 観点 | 現状 | 改善の余地 |
|------|------|------------|
| **時系列 TE** | その行より前のみで集計。val/te は tr のカテゴリ別スムージング平均でマッピング。 | リークの心配は小さい。 |
| **正規化** | 02 の掛け合わせでは release_decade_norm, review_year_norm 等を **train の min/max のみ**で計算。 | 妥当。特に対応不要。 |
| **directors / authors** | 01・02 で「category のまま。TE や頻度を足すと悪化」と結論。 | 方針どおりそのままでよい。監督・著者の「人数」（カンマ数など）を 1 列だけ足す案は、TE/頻度ではないので試す価値はあるが必須ではない。 |

---

## 3. 次に試すべきこと（優先度付き）

### 優先度 高（すぐ効く・穴埋め）

1. **段階的組み合わせの実行と 02 への反映**  
   train_feature_engineering の「段階的組み合わせ実験」セルを実行し、stage0〜9 の CV_AUC 表を **02_FEATURE_ENGINEERING.md §4** に転記する。どの段階まで足すと良くて、どこから過学習気味になるかを残す。

2. **ベースラインへの「改善 1 列」の取り込み**  
   最も効いた **p03（critic_te_x_genre_Documentary）** を、train_baseline の FEATURES に 1 列追加して CV を回し、提出用パイプラインを「38 列 → 39 列」にするかどうか決める。比較用に「38 列のまま」の結果も 1 行残すとよい。

### 優先度 中（試す価値あり）

3. **publisher_name_freq の追加**  
   01 で +0.0003 だった。既存列は残したまま 1 列足すだけなので、train_baseline または train_feature_engineering で 1 設定として追加し、CV で 38（または 39）列と比較する。

4. **テキスト lda_10 の 1 設定追加**  
   movie_info に LDA(10) を**追加**した設定を、train_baseline か train_feature_engineering で 1 本だけ追加し、「38 列 vs 38＋lda_10」で CV 比較する。テキストは微増だが、ベースラインに未取り込みなので 1 回は試す価値あり。

5. **3C3 の共通化**  
   時系列 TE とビンの計算を experiment_encodings などの共通モジュールに切り出し、train_baseline と train_feature_engineering の両方から呼ぶ。コードの重複削減と変更漏れ防止用。

### 優先度 低（余裕があれば）

6. **ハイパーパラメータの探索**  
   特徴量を 38＋α で固めたあと、LGB の n_estimators / learning_rate / num_leaves などを optuna 等で探索する。

7. **テキストベクトル実験の再開**  
   nmf_20 以降（concat_mi_title_* 等）を train_text_vector_experiments の再開用で実行。テキストは伸びが小さいため、優先度は低くてよい。

8. **監督・著者まわりの「足す」案**  
   directors / authors は TE・頻度はやらない。代わりに「監督が複数いるか」（カンマ数などから 0/1 や本数）を 1 列だけ足す案は、方針に反しないので余裕があれば試せる。

---

## 4. まとめ

- **実装でまだ改善できそうなところ**: 特徴量定義の分散、3C3 の重複、**未取り込みの効いた追加**（publisher_name_freq, lda_10, 改善 9 パターン）、**段階的組み合わせ結果の未記載**、LGB の未チューニング。
- **次にやるとよいこと**: (1) 段階的組み合わせの実行と 02 への結果転記、(2) ベースラインへの p03 の 1 列追加と CV 比較、(3) publisher_name_freq と lda_10 の 1 設定追加、の順で進めると、漏れがなくかつ 01・02 の知見を活かせる。
