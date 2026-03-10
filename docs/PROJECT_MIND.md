# コンペ方針・結果・今後のやること（1本まとめ）

失敗したこと・成功したこと・今後の方針を1本に集約。**方針: テキストの embedding + ベースライン + 上位解法を試す。**

---

## 1. 成功したこと

| 方針 | Public Score | 備考 |
|------|--------------|------|
| ベースライン38のみ | 0.75097 | 従来の軸。 |
| **ベースライン38 + movie_info の embedding（8次元）** | **0.75391** | **現時点で最高。** テキストのベクトル化が有効。 |
| ベースライン38 + embedding（16次元） | 0.75314 | 8次元よりやや低いがベースラインより上。 |
| ベースライン38で既知/未知2本 | 0.75079 | ベースラインにほぼ並ぶ。 |

- **テキストの embedding**: `movie_info` を OpenAI `text-embedding-3-small` で映画単位に1回だけベクトル化 → 保存（`outputs/embeddings/movie_info_embeddings.pkl`）→ PCA で 8 または 16 次元にして特徴量に追加。v2 は足さず **ベースライン + embedding だけ** で伸びた。

---

## 2. 失敗したこと

| 方針 | 結果 | 備考 |
|------|------|------|
| v2（OOF TE+カウント）のみ | 0.73743 | ベースラインより大幅に低い。 |
| v2 + 比率 / NMF / SVD / v2_all | いずれも 0.73〜0.74 台 | 特徴を足すと過学習気味。 |
| **v2 + OpenAI embedding（16次元）** | **下がった** | ベースライン+embedding より悪化。v2 を embedding に足すのはやめる。 |
| 既知/未知を v2 や v2_all で | スコア落ち | ベースライン基準の既知/未知のみ並ぶ程度。 |

→ このコンペでは **「ベースライン + テキストのベクトル化」に絞る** のが有効。v2 系をそのまま足すと悪化する。

---

## 3. 今後の方針

1. **軸**: **テキストの embedding + ベースライン**  
   - メイン提出は ベースライン38 + movie_info の embedding（PCA 8次元）。v2 は足さない。

2. **試す**: **上位解法を「ベースライン+embedding の上で」試す**  
   - 既知/未知の二モデル（ベースライン+embedding で既知用・未知用を分ける）、シードアベレージング、タイトル+movie_info の連結 embedding、次元数・PCA の微調整など。  
   - 締め切り直前に余裕があれば: アンサンブル・別モデル。

3. **やらない**: v2 / v2_all を embedding に足す。既知/未知を v2 で再挑戦する。

---

## 4. 実行の仕方（最小限）

### テキストの embedding（1回だけ）

- API キー: `config/openai_api_key.txt` の1行目に `sk-` で始まるキーを貼る。環境変数 `OPENAI_API_KEY` でも可。
- コスト見積もり（API は叩かない）: `python archive/run_openai_embeddings_once.py --estimate`
- 実行: `python archive/run_openai_embeddings_once.py`（プロジェクトルートで）  
  → 映画単位で `movie_info` をベクトル化し、`outputs/embeddings/movie_info_embeddings.pkl`（または .parquet）に保存。以降はこのファイルを使い回す。

### 提出用 CSV（ベースライン + embedding）

- 3パターン一括: `python archive/run_openai_three_submissions.py`（プロジェクトルートで）  
  - pattern1: ベースライン38 + OpenAI 16次元 → 0.75314  
  - **pattern2: ベースライン38 + OpenAI 8次元 → 0.75391（推奨）**  
  - pattern3: v2 + OpenAI 16次元 → 使わない（下がった）

- 単体1本（pattern2 相当）: `python archive/run_baseline_openai_submission.py`

### 用語

- **ベースライン38**: `lib.BASELINE_FEATURES` の38列（時系列TE・ビン・テキストメタ等）。LGB は `lib.BASELINE_LGB_PARAMS`。
- **v2**: ベースライン38 + OOF TE 6列 + 批評家/映画のレビュー数。このコンペでは単体・embedding 併用ともスコアが落ちている。
- **上位解法**: atmaCup #15 等の OOF TE・比率・NMF・SVD・既知/未知二モデル・シードアベレージング。実装は `archive/top_solutions.py`。**ベースライン+embedding を土台にした上で** 試す。

---

## 5. 主なファイル

| ファイル | 役割 |
|----------|------|
| `train_baseline.ipynb` | メインのノートブック（ルートに1本だけ）。ベースライン学習・CV・提出。 |
| `lib/pipeline.py` | `get_baseline_data`, `BASELINE_FEATURES`, `BASELINE_LGB_PARAMS` |
| `lib/openai_embeddings.py` | embedding の取得・保存・読み込み（`load_movie_info_embeddings`。特徴量追加は `archive/top_solutions.add_openai_embedding_features`） |
| `archive/run_openai_embeddings_once.py` | 1回だけ API で embedding を保存（ルートで `python archive/run_openai_embeddings_once.py`） |
| `archive/run_openai_three_submissions.py` | ベースライン+embedding の3パターン提出を一括作成 |
| `archive/top_solutions.py` | 上位解法（OOF TE・NMF・SVD・既知/未知・シードアベレージ等）と `add_openai_embedding_features` |

---

## 6. 関連ドキュメント（残しているもの）

- `docs/01_PREPROCESS.md`, `02_FEATURE_ENGINEERING.md`, `03_ANALYSIS.md`, `04_METRICS_GUIDE.md` — 前処理・特徴量・分析・指標の説明。

---

## 7. 類似コンペ・上位解法からのヒントメモ

ここでは、Rotten Tomatoes / 映画レビュー系の Kaggle コンペ・GitHub リポジトリ・論文から、「このコンペにそのまま/一部流用できそう」と思ったアイデアだけをメモする。  
実装するかどうかは、**「ベースライン + OpenAI embedding」軸との相性**と、締切までの時間を見て判断する。

- **(A) 古典NLP系（TF-IDF + 線形/GBDT）**
  - Kaggle の `Sentiment Analysis on Movie Reviews`（Rotten Tomatoes フレーズ分類）では、TF-IDF + Logistic Regression / Linear SVM / LightGBM といった古典構成が今でも強い。
  - 多くの上位ノートは「**TF-IDF(1-2gram) → 次元圧縮 or 正則化強めで線形モデル**」を採用しており、RNN より安定してスコアを出している例も多い。
  - **このコンペへの落とし込み案**
    - `movie_info` / `movie_title + movie_info` を TF-IDF → TruncatedSVD で 20〜50 次元に圧縮し、今の OpenAI embedding と同じように **「映画単位のテキストベクトル」** として LightGBM に入れる。
    - すでに `docs/01_PREPROCESS.md` で TF-IDF + SVD の実験を少しやっているが、「**embedding + TF-IDF-SVD の共存**」はまだ本格的には試していないので、アンサンブル候補として残す。

- **(B) Transformer / BERT 系の fine-tuning**
  - IMDB / Rotten Tomatoes のレビュー分類タスクでは、`bert-base-uncased` / `distilbert-base-uncased` をそのまま fine-tuning しただけで、TF-IDF ベースをかなり上回るのが定番になっている。
  - 論文・Kaggle ノートでは、**BERT で文埋め込み → そのまま線形層** だけでなく、「**BERT から CLS 埋め込みを抜き出して LightGBM/ロジスティック回帰に入れる**」ハイブリッド構成も使われている。
  - **このコンペへの落とし込み案**
    - `movie_info` / タイトル+映画情報を BERT/DistilBERT でエンコードし、**映画単位で 768 次元程度の埋め込み**を事前計算して保存（今の OpenAI embedding と同じ運用）。
    - 直接 fine-tuning して end-to-end 学習するのではなく、「**BERT 埋め込みを作って LightGBM に渡す**」形にすれば、今のパイプラインに近い形で導入できる。
    - ただし計算コスト・実装コストが重いため、「締切直前の追加ブースト候補」としてメモだけ残す。

- **(C) 埋め込み + GBDT のハイブリッド**
  - いくつかの研究・実装（例: DistilBERT 埋め込み + LightGBM）では、「**深層モデルで埋め込みだけ作り、決定木系モデルで最終分類**」という構成を採用している。
  - これは「非線形なテキスト表現」と「カテゴリ/数値特徴をうまく扱う GBDT」の長所を組み合わせたもの。
  - **このコンペですでにやっていること**
    - OpenAI embedding（text-embedding-3-small / -3-large）を映画単位で作成し、LightGBM に入れているのはまさにこの構成。
    - 追加でやれるとしたら、「**embedding 次元数の調整（PCA 8 / 16 / 32）」「small / large のアンサンブル」など。

- **(D) マルチビュー特徴（レビュー側 / 映画側を分ける）**
  - 一部の Rotten Tomatoes 系実装では、「**レビュー本文からの特徴**」と「**映画メタ情報（ジャンル・監督・公開年など）からの特徴**」を別々に作り、最後に結合して学習している。
  - **このコンペへの落とし込み案**
    - 既に「映画単位 embedding（movie_info, タイトル+映画情報）」を作っているので、これを**映画側ビュー**として扱い、レビューレベルの長さ/日付/批評家特徴と組み合わせる。
    - 余裕があれば、「**批評家コメント本文（quote 列などがあれば）をレビュー側 embedding にする**」ことで、映画ビュー + レビュービューの2本立てにする。

- **(E) アンサンブル・シードアベレージング**
  - IMDB / Rotten Tomatoes 系の多くの上位解法では、**異なるシード・異なるモデル（TF-IDF + LR, embedding + GBDT, BERT 単体など）をアンサンブル**してスコアを底上げしている。
  - すでに atmaCup 系上位解法の「シードアベレージ」アイデアは `archive/top_solutions.py` に取り込んでいるので、
    - OpenAI embedding（small / large, 次元違い）を使った複数モデル、
    - 余裕があれば TF-IDF-SVD モデルや BERT 埋め込みモデル
    を混ぜて **最終提出だけアンサンブル** する余地がある。

→ まとめると、**「映画単位のテキスト埋め込み + GBDT」** という今の軸は、類似タスクの上位解法と方向性が合っている。  
残り時間とコストを見ながら、

1. OpenAI embedding small / large や次元数のパターンを増やしてシードアベレージ/アンサンブル  
2. 余裕があれば TF-IDF-SVD や BERT 埋め込みを「もう1本のビュー」として追加

という順で試すのが現実的なロードマップ。

---

## 8. Grade 3 レベルで試したい手法メモ（疑似ラベル・AWP・OpenAI×HuggingFace）

上位解法・NLP コンペでよく使われる「攻め」のテクニックをメモ。**特に疑似ラベル（1）は、このコンペの「文脈が効く映画レビュー」と相性が良く、試す価値が高い。**

### 8.1 疑似ラベル（Pseudo-Labeling）— 攻めの半教師あり

映画レビューのように文脈が重要なデータでは、モデルが「自信満々に間違える」こともある。Grade 3 レベルでやるなら次のステップを踏む。

1. **確信度の閾値**
   - 予測確率が **0.95 以上** または **0.05 以下** のものだけをテストから抽出し、擬似正解として学習に追加する。
   - 閾値を緩めるとノイズが増えるので、最初は厳しめ（0.95/0.05）で試す。

2. **ソフトラベルの検討**
   - 0/1 に決めつけず、**予測確率（例: 0.98）をそのままラベルとして使う**手法も有効。
   - 「絶対ではないが、かなり高評価っぽい」というニュアンスをモデルに学習させられる。LightGBM では `sigmoid` 出力をそのまま重みやターゲットの連続値として扱う形で組み込める。

3. **不均衡の修正**
   - テストから取った擬似ラベルが「高評価ばかり」に偏っていないか確認する。
   - 偏っている場合は、**サンプリングで Fresh/Rotten のバランスを整える**のがセオリー（例: 少ない方に合わせてダウンサンプル、または重みで調整）。

→ **このコンペでは「ベースライン + embedding」で一度 test を予測し、高確信度のみ擬似ラベル化して再学習するパイプラインを組むとおもしろい。** 実装優先度: 高。

---

### 8.2 AWP（Adversarial Weight Perturbation）— 過学習をねじ伏せる

NLP コンペ（とくにデータが数千件程度）では、**AWP は「必須級」**と言われることが多い。

- 通常の学習は「損失の谷」を探すが、AWP は **「谷の底が平らな場所」**（フラットなミニマム）を探す。
- わずかな入力の違い（ノイズ）で予測がひっくり返るのを防ぎ、**テストへの汎化**が期待できる。

**実装のコツ**

- 学習の **後半エポックから** AWP を導入するのが一般的。最初から入れると学習が進まないことがある。

**アピール**

- 「精度だけでなく、実運用時の頑健性（ロバスト性）を考慮して AWP を採用した」と説明できると、コンサル・レビュー的にも評価されやすい。

※ 現在のパイプラインは **LightGBM のみ**なので、AWP をそのまま使うには **ニューラルネット（PyTorch 等）で embedding を入力にした分類モデル**を別途用意する必要がある。NN を導入する場合の候補としてメモ。

---

### 8.3 アンサンブル: OpenAI × HuggingFace のハイブリッド

- **OpenAI text-embedding-3-large**: 汎用的な意味を捉えるのが得意。
- **映画レビュー特有の「皮肉」「期待外れ」**（期待していたのにダメ＝低評価）などは、**感情分析特化モデル**を混ぜると強い。

**おすすめの組み合わせ例**

| Feature | モデル | 次元目安 |
|--------|--------|----------|
| Feature 1 | OpenAI `text-embedding-3-large` | 3072 |
| Feature 2 | HuggingFace `cardiffnlp/twitter-roberta-base-sentiment-latest`（SNS/レビュー向け） | 768 など |

**統合方法**

- 上記2つを **結合（Concat）** し、その上に **LightGBM や CatBoost** で学習させる「スタッキング」が、スコアが出やすい典型パターン。
- 映画情報・タイトル+映画情報は OpenAI で既に取得済み。HuggingFace 側は **同じテキスト（movie_info や タイトル+映画情報）を RoBERTa でエンコード**し、映画単位で 1 ベクトルにしてから concat すれば、今のパイプラインに載せやすい。

→ 実装する場合は、まず OpenAI large embedding のみで安定させてから、RoBERTa 埋め込みを「追加の列」として足す形が無難。

---

以上を **テキストの embedding + ベースライン + 上位解法を試す** 方針の外部知見メモとする。
