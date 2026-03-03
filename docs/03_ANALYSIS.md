# 分析結果 — 現状の改善余地と次に試すこと

01_PREPROCESS.md と 02_FEATURE_ENGINEERING.md を踏まえ、**現状実装でまだ改善できそうなところ**を洗い出し、**次に試すべきこと**を優先度付きでまとめる。

※ ノートの実行結果（提出用・分析用・実験用 CSV）はすべて `outputs/` 以下（`submissions/`, `analysis/`, `experiments/`）に保存される。

---

## 1. 現状サマリ（01・02 から）

| 項目 | 内容 |
|------|------|
| **ベース** | 38 列（create_features の日付・ジャンル・カテゴリ ＋ ノート内の 3C3 ＋ movie_info_meta ＋ movie_title 長さ・語数）。CV_AUC 約 **0.7602**。 |
| **効いたこと** | 時系列 TE 追加（critic_name, production_company）。ビン追加（runtime_bin, movie_age_bin, release_decade）。既存を残して「足す」だけ。 |
| **効かなかったこと** | 8 列の TE/頻度置き換え、directors/authors の TE・LOO・頻度、movie_title の頻度置き換え、critic_name の LOO。 |
| **テキスト** | movie_info の TF-IDF/LDA 等は +0.0001〜0.0006 の微増。lda_10 が最大 +0.0006。 |
| **掛け合わせ** | 15 パターン中 9 パターンが CV で改善。ベスト単体は p03 で +0.0010。段階的組み合わせ（stage0〜9）をパブリックで検証したところ **全ステージでベースよりスコア低下 → 全部ボツ、採用見送り**（02 §4.2）。 |

---

## 2. 実装のまだ改善できそうなところ（洗い出し）

### 2.1 パイプライン・コードまわり

| 観点 | 現状 | 改善の余地 |
|------|------|------------|
| **特徴量定義の分散** | 38 列のうち、create_features は日付・ジャンル・content_rating, publisher_name のみ。3C3・テキストメタ・カテゴリ 8 列の扱いは **train_baseline / train_feature_engineering のノート内** に直書き。feature_engineering/features.py の FEATURES は 19 列で、ノートの 38 列と一致していない。 | 運用ベースを「ノートの FEATURES リスト」に揃えたまま、**どこまでを create_features / lib.encodings に寄せるか** を整理すると、再現や Kaggle 提出時のミスが減る。必須ではない。 |
| **3C3 の再利用** | 時系列 TE とビンは train_baseline / train_feature_engineering のノートに同じコードが 2 回書かれている。 | lib.encodings に「3C3 用の時系列 TE ＋ ビン」を関数化して呼び出すと、ノートが短くなり変更漏れを防げる。 |
| **欠損処理** | runtime は train の中央値で埋め。movie_age_days の負値は NaN にしている。 | 01 では欠損フラグは「変化なし」。現状の欠損処理で十分。特に対応不要。 |

### 2.2 まだ取り込んでいない「効いた」追加

| 候補 | 01・02 での結果 | 現状 | 改善の余地 |
|------|-----------------|------|------------|
| **publisher_name_freq** | 01: category は残したまま出現回数を**追加**で +0.0003。 | ベース 38 列には入っていない。 | 1 列足すだけなので、**ノートで追加して CV 比較**すれば取り込み可否を判断できる。 |
| **テキストベクトル lda_10** | 01: movie_info に LDA(10) を**追加**で +0.0006（テキスト系で最大）。 | ベースはテキストメタ（長さ・語数）のみ。TF-IDF/LDA は train_text_vector_experiments で比較したが、ベースライン側には組み込んでいない。 | ベース 38 ＋ lda_10 を **train_baseline や train_feature_engineering の 1 設定として追加**し、CV で 38 のみと比較する価値あり。 |
| **改善 9 パターンの採用** | 02: p03 単体で CV +0.0010。段階的組み合わせで stage0〜9 の提出をパブリック検証済み。 | **パブリックでは全ステージでベースよりスコア低下のため採用見送り**（02 §4.2）。 | 9 パターンの段階的追加はボツ。ベースは 38 列のまま。別軸（publisher_name_freq, lda_10 等）やハイパラ探索を検討。 |

### 2.3 実験・検証まわり

| 観点 | 現状 | 改善の余地 |
|------|------|------------|
| **段階的組み合わせの結果** | CV 結果は 02 §4.1 に転記済み。stage0〜9 の提出をパブリックで検証し、**全パターンでベースよりスコア低下**。02 §4.2 に結論（全部ボツ）を記載済み。 | 特になし。 |
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

## 3. 予測結果（当たり/外れ）から見えたこと

`train_with_predictions.csv` と `prediction_summary_by_*.csv` をもとに、モデルがどこで外しているかを整理する。

### 3.1 全体傾向

- 時系列 fold の accuracy はおおむね **0.719〜0.721** で安定。
- ただし year ごとの AUC は差があり、**2015 年が最弱（AUC=0.7512）**。
- 予測ラベルの傾向として、全 year で **pred_pos_rate が pos_rate より高い**（例: 2016 は 0.7686 vs 0.6604）。
- その結果、**FP_rate が高め（約 0.54〜0.57）**、FN_rate は比較的低い（約 0.13）という「陽性寄り」な判定傾向が出ている。

### 3.2 セグメント別の弱点

- **review_year**: 2015 年の AUC が最低（0.7512）。年ドリフトの影響候補。
- **content_rating**: NC17 が最低（AUC=0.7312, n=240）。サンプルが少なく不安定だが弱点候補。
- **genre_Documentary=1**: AUC=0.7149（n=14,931）で明確に弱い。
- **top_critic=True**: AUC=0.7467（n=54,956）で False より弱い。

### 3.3 重要度から見た示唆

- 上位は `directors`, `authors`, `rotten_tomatoes_link`, `movie_title`, `critic_name` など高カード列が中心。
- 一方で 0 重要度の列が多い（`runtime_bin`, `movie_age_bin`, `release_decade`, 複数 genre など）。
- 現状の 3C3 の一部やビン列は、ベースラインでは寄与が薄い可能性がある。

### 3.4 `train_metric_driven_experiments.ipynb` で試すこと

このノートは、**「AUC だけでなく弱点セグメントと誤差バランスを同時に改善できるか」** を検証するための実験ノート。

- ベースは `lib.get_baseline_data()` の 38 列で固定し、そこに **少数の追加列**だけを足して比較する。
- 比較する設定は次の 6 つ（`EXPERIMENT_CONFIGS`）:
  - `base`
  - `publisher_freq`（publisher の出現頻度）
  - `doc_x_critic_te`（`genre_Documentary * critic_name_te_ts`）
  - `year_norm_x_critic_te`（年正規化 × critic TE）
  - `topcritic_x_critic_te`（top_critic × critic TE）
  - `publisher_freq__doc_x_critic_te`（上記 2 つの同時追加）
- 各設定で時系列CVを回し、`metric_driven_experiment_results.csv` に次を保存:
  - 全体性能: `auc_mean`, `auc_std`, `logloss_mean`, `brier_mean`
  - 誤差バランス: `fp_rate_mean`, `fn_rate_mean`, `pos_gap_mean`
  - 弱点セグメント: `auc_review_2015`, `auc_doc_1`, `auc_topcritic_true`
- 目的は「全体AUCの微増」だけでなく、**2015 / Documentary / top_critic=True の弱点が改善しているか** を同時に見ること。

---

## 4. 次に試すべきこと（誤差分析ベース、優先度付き）

### 優先度 高（まずやる）

1. **2015 年ドリフト対策の時系列特徴を 1〜2 列だけ追加**
   - 候補: `review_year` と TE の交互作用（既に試した p15 系を単独で再検証）、`review_month` 系の時系列補助列。
   - 目的: 2015 年の順位付け悪化（AUC 低下）を改善。

2. **Documentary 向けの軽量な追加列を 1 本だけ検証**
   - `genre_Documentary` が弱点なので、`genre_Documentary` と既存強特徴（`critic_name_te_ts` など）の交互作用を 1 列だけ試す。
   - ただし大量追加は避ける（過去に段階追加でパブリック悪化済み）。

3. **予測確率の校正状況を定点観測**
   - AUC 自体は閾値非依存だが、現状は陽性寄り（FP 多め）なので、改善判定用に `pred_pos_rate - pos_rate` を毎回記録する。
   - モデル比較時に「AUC + バイアス指標（FP/FN バランス）」を併用する。

### 優先度 中

4. **publisher_name_freq の追加（再検証）**
   - 01 で微増実績あり。高カード由来の情報補完として再検証価値が高い。

5. **lda_10 の追加（再検証）**
   - テキスト系の中では最も改善幅があった設定。Documentary 弱点の補完を期待して 1 設定で検証。

6. **LightGBM ハイパラ探索（小さく）**
   - `num_leaves`, `min_child_samples`, `feature_fraction`, `lambda_l1/l2` を狭い範囲で探索し、過学習気味の特徴（CV 良化→Public 悪化）を抑える。

### 優先度 低

7. **0 重要度列の棚卸し**
   - 0 重要度列を一時的に外した「圧縮ベース」を作り、CV と Public を比較（ノイズ削減の確認）。

8. **監督・著者の軽量派生**
   - TE/頻度ではなく、人数など低リーク・低自由度の派生のみ少数試す。

---

## 5. まとめ

- **実施済み・結論**: 段階的組み合わせ（stage0〜9）はパブリックで全滅（ベース割れ）なので採用しない。
- **新たな知見**: 弱点は `review_year=2015`, `genre_Documentary=1`, `top_critic=True`。また全体に陽性寄り判定で FP が多い。
- **次の方針**: 「弱点セグメントに効く列を少数追加」＋「過学習を抑える調整」を行い、毎回 `train_with_predictions.csv` 系で誤差構造を追跡する。
