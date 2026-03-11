# コンペ方針・結果・今後のやること（1本まとめ）

失敗したこと・成功したこと・今後の方針を1本に集約。**方針: テキストの embedding + ベースライン + 上位解法を試す。**

---

## 1. 成功したこと

| 方針 | Public Score | 備考 |
|------|--------------|------|
| ベースライン38のみ | 0.75097 | 従来の軸。 |
| **ベースライン38 + movie_title_info（small）embedding + PCA 16** | **0.75449** | **現時点で最高。** タイトル+映画情報の small が最良。 |
| ベースライン38 + movie_info の embedding（8次元） | 0.75391 | 映画情報のみでも有効。 |
| ベースライン38 + embedding（16次元） | 0.75314 | 8次元よりやや低いがベースラインより上。 |
| ベースライン38で既知/未知2本 | 0.75079 | ベースラインにほぼ並ぶ。 |

- **テキストの embedding**: 4 種類（movie_info / movie_info_large / movie_title_info / movie_title_info_large）× PCA 8/16 で提出比較した結果、**movie_title_info（small）+ PCA 16** が Public 最高（0.75449）。詳細は `docs/05_EMBEDDING_SUBMISSION_RESULTS.md`。
- この設定を基準に、次元削減を PCA 以外（SVD, ICA, NMF, UMAP 等）に変えた 8 パターンを `train_embedding_submission.ipynb` で試す。

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

### 2.5 Embedding 種別の比較結果（時系列CV と Public 提出）

**時系列CV（train_baseline_embedding_experiments / run_embedding_experiments）**

- **movie_info**（あらすじのみ）: 時系列CV 0.7651（PCA8）, 0.7638（PCA16）、GroupKFold 0.7384 / 0.7380。
- **movie_info_large**: 時系列CV 0.7638〜0.7640、GroupKFold 0.7378〜0.7381。
- **movie_title_info**（タイトル + 映画情報・small）: 時系列CV 0.7642〜0.7658、GroupKFold 0.7388〜0.7395。
- **movie_title_info_large**: 時系列CV 0.7646〜0.7640、GroupKFold 0.7389〜0.7378。

**Public 提出での比較（8 本提出）**

| Embedding | PCA 8 | PCA 16 |
|-----------|-------|--------|
| movie_title_info（small） | 0.75437 | **0.75449** |
| movie_title_info_large | 0.75388 | 0.75440 |
| movie_info | 0.75394 | 0.75237 |
| movie_info_large | 0.75317 | 0.75370 |

**結論**: **Public では movie_title_info（small）+ PCA 16 が最高（0.75449）**。これを基準に、次元削減手法を 8 パターン（PCA / SVD / ICA / NMF / UMAP 等）変えた実験を `train_embedding_submission.ipynb` で実施。結果は `docs/05_EMBEDDING_SUBMISSION_RESULTS.md` にまとめる。

---

## 3. 今後の方針

1. **軸**: **テキストの embedding + ベースライン**  
   - メイン提出は ベースライン38 + **movie_title_info（small）** の embedding（**PCA 16 次元**）。Public 0.75449。v2 は足さない。

2. **試す**: **次元削減 8 パターン + 上位解法**  
   - **次元削減**: movie_title_info 固定で PCA 以外（TruncatedSVD, ICA, NMF, UMAP 等）を試す → `train_embedding_submission.ipynb`。結果は `docs/05_EMBEDDING_SUBMISSION_RESULTS.md`。  
   - 既知/未知の二モデル、シードアベレージング、次元数の微調整。  
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

- **推奨（Public 最高）**: `train_embedding_submission.ipynb` で **movie_title_info + PCA 16** を選択して実行 → **0.75449**。次元削減を SVD/ICA/NMF/UMAP 等に変えた 8 パターンも同ノートで一括作成可能。
- 従来スクリプト: `python archive/run_openai_three_submissions.py`（movie_info ベース）  
  - pattern2: ベースライン38 + OpenAI 8次元 → 0.75391
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
| **`train_embedding_submission.ipynb`** | **最良 embedding（movie_title_info）固定で次元削減 8 パターンの提出 CSV を一括作成。** 結果は `docs/05_EMBEDDING_SUBMISSION_RESULTS.md`。 |
| `lib/embedding_reduction.py` | 次元削減 8 手法（PCA / SVD / ICA / NMF / UMAP 等）の fit/transform ラッパー。 |
| `lib/pipeline.py` | `get_baseline_data`, `BASELINE_FEATURES`, `BASELINE_LGB_PARAMS` |
| `lib/openai_embeddings.py` | embedding の取得・保存・読み込み（`load_movie_info_embeddings`。特徴量追加は `archive/top_solutions.add_openai_embedding_features`） |
| `archive/run_openai_embeddings_once.py` | 1回だけ API で embedding を保存（ルートで `python archive/run_openai_embeddings_once.py`） |
| `archive/run_openai_three_submissions.py` | ベースライン+embedding の3パターン提出を一括作成 |
| `archive/top_solutions.py` | 上位解法（OOF TE・NMF・SVD・既知/未知・シードアベレージ等）と `add_openai_embedding_features` |

---

## 6. 関連ドキュメント（残しているもの）

- `docs/01_PREPROCESS.md`, `02_FEATURE_ENGINEERING.md`, `03_ANALYSIS.md`, `04_METRICS_GUIDE.md` — 前処理・特徴量・分析・指標の説明。
- **`docs/05_EMBEDDING_SUBMISSION_RESULTS.md`** — Embedding 種別×PCA の Public 比較と、次元削減 8 パターン・追加実験の結果メモ。提出用ノートは `train_embedding_submission.ipynb`。
- **`docs/06_EMBEDDING_REDUCTION_TECHNOLOGY.md`** — 次元削減 8 手法＋追加実験 4 パターンの技術解説（特徴・数理・使い分け）。

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

- **(F) atmaCup #15 3位解法（maruyama）— 転用しやすいポイント**
  - **タスク**: アニメのユーザー×作品の評価予測（回帰）。私たちのコンペは「批評家×映画の Fresh/Rotten 予測」で、**ユーザー↔批評家・作品↔映画**の対応で考えられる。
  - **参考になる点（モデルアーキテクチャ以外で使えるところ）**
    | 3位のやり方 | 私たちへの落とし込み |
    |-------------|----------------------|
    | **NMF を「行×列」の評価行列」で複数カテゴリ・複数次元** | 批評家×映画、批評家×ジャンル、批評家×制作会社など**複数の行列**を NMF 分解し、**次元数 5/10/20/50 など複数パターン**を特徴量にする。`archive/top_solutions.py` の NMF は 1 種類なので、「同じ NMF で n_components を 16/32/64 と変えた複数本」を足す。 |
    | **CV を「テストと同様に未知ユーザーが 23%」になるよう分割** | テストで未知映画が一定割合あるなら、**検証でも「学習に登場しない映画」を一定割合入れる**。既存の **GroupKFold（group=映画）** や **既知/未知二モデル** がまさにそれ。3位は KFold と GroupKFold を組み合わせた UnknownUserKFold で再現。 |
    | **feature_fraction を小さめ + 学習率小さめでゆっくり学習** | 特徴量が多いときの過学習抑制。私たちも **feature_fraction=0.7 前後、learning_rate=0.01〜0.05** を 1 本試す価値あり（現在は 0.04 前後）。 |
    | **アンサンブル: 次元数違い・シード違い・「全データで学習した 1 本」を足す** | **NMF/embedding の次元数違い**（PCA 8 / 16 / 32）、**シード違い**、**CV の各 fold モデル + 全 train で学習した 1 本**の予測を平均。3位は「学習データ全件で学習させたモデルを加えると Public が上がった」と明言。 |
  - **私たちですでにやっていること**: NMF・SVD（2位/3位風）、既知/未知二モデル、シードアベレージングは `archive/top_solutions.py` にあり。**まだ足していない**のは「NMF/embedding の**次元数バリエーションを複数本アンサンブル**」「**全データ 1 本をアンサンブルに含める**」あたり。

- **(G) atmaCup #15 6位解法（Shun_PI）— 類似度を使った Target Encoding の応用**
  - **考え方**: 普通の TE は「そのユーザーが付けた他作品の平均」「その作品に他ユーザーが付けた平均」だが、**似ていないユーザー・作品も平均に混ざってしまう**。→ **「予測対象と似ているユーザー／作品」だけを top-k で絞り、その k 件だけの平均を TE として使う**と、より予測に近い値になる。
  - **類似度の種類**
    | 種類 | 説明 | リーク・実装 |
    |------|------|--------------|
    | **Implicit（視聴したかどうか）** | 「どの作品を視聴したか」の一致度で類似度。**正解ラベルを使わない**のでリークしにくい。 | Jaccard（視聴集合の交わり/結び）、Hamming（視聴 0/1 ベクトルの一致）、Word2Vec（「作品 X を視聴したユーザー集合」を文と見てユーザー埋め込み→コサイン類似度）。6位はこれだけで Private 13 位相当。 |
    | **Explicit（スコアの付け方）** | 「共通で評価した作品でのスコア差」で類似度。**正解を使うのでリークに注意**。時系列なら「その行より前」だけで類似度・TE を計算。 | L1/L2 を「共通評価作品のみ」で計算。問題: 共通が少ないと距離 0、多いと距離が大きくなりがち。→ **各作品の寄与から全体平均を引き**、必要なら **sqrt(共通数) で割る**と、類似度の順序が妥当になる。6位はこれで Private 9 位→アンサンブルで 6 位。 |
  - **特徴量の作り方（共通）**
    - 類似度を複数種類 × **k = [1, 3, 5, 10, 50, 100, …]** の各 k について、(1) **予測したい批評家に似た批評家 top-k が、その映画に付けた Fresh 率の平均**、(2) **予測したい映画に似た映画 top-k に、その批評家が付けた Fresh 率の平均**、の 2 本を特徴量にする。k を変えると 1 種類あたり 2×len(k) 列になる。
  - **私たちへの落とし込み**
    - **批評家↔ユーザー、映画↔作品**の対応で考える。
    - **Implicit**: 批評家ごと「レビューした映画の集合」→ 批評家間 Jaccard / Hamming。映画ごと「レビューした批評家の集合」→ 映画間 Jaccard / Hamming。または「映画 X をレビューした批評家」を文として Word2Vec → 批評家埋め込み → コサイン類似度。**時系列を守るなら「その行より前のレビューだけ」で集合・埋め込みを構成**する。
    - **Explicit**: 批評家 u と v の類似度 = 共通でレビューした映画での Fresh/Rotten (0/1) の L1。同様に映画間も「共通でレビューした批評家での 0/1 の L1」。**リーク防止**: 予測行の (批評家, 映画) のラベルを平均に含めない＋類似度計算からも除外。6位は「寄与を前計算し、総和から今の映画の寄与を引く」で高速化している。
  - **注意**: 実装コスト・計算量は大きい。まず **Implicit の Jaccard だけ**で「似た批評家 top-k のその映画の Fresh 率」「似た映画 top-k のその批評家の Fresh 率」を 1〜2 個の k で試し、CV が伸びるか見るのが現実的。

- **(H) atmaCup #15 5位解法（statsu）— Seen/Unseen 分割・グラフ埋め込み・スコア重み**
  - **スコア**: CV 1.1582, Public 1.1754, Private 1.1409。Seen と Unseen でモデルを分け、CV は 0.73×seen_rmse + 0.23×unseen_rmse でテスト比率に合わせている。
  - **Seen / Unseen の扱い**
    | 5位のやり方 | 私たちへの落とし込み |
    |-------------|----------------------|
    | **学習** | Seen 用と Unseen 用で**別々に学習**。特徴量の種類は同じだが、Unseen 用では「Unseen 側の正解ラベルを特徴量計算に使わない」。 | 既知映画用・未知映画用の二モデル（`archive/top_solutions.py`）と同様。未知用では **未知映画の行の target を NaN 扱いにして TE 等の計算に含めない** ようにする、という設計が 5 位と一致。 |
    | **CV（Unseen 用）** | GroupKFold で train / unseen_val に分割。train をさらに GroupShuffleSplit で **seen_train / unseen_train** に分割。**unseen_train の score は特徴量計算には使わず、学習の正解ラベルとしては使う**。5 fold。 | 未知映画モデルで「学習データのうち未知映画の行は TE 計算から除外」するのと同じ思想。検証は「学習に登場しない映画（GroupKFold）」で行う。 |
    | **CV スコア** | 0.73×seen_rmse + 0.23×unseen_rmse でテストの seen/unseen 比率に合わせる。 | テストで未知映画が約 2〜3 割なら、**0.7×時系列CV_AUC + 0.3×GroupKFold_AUC** のような加重で 1 本の CV 指標にしてもよい。 |
  - **特徴量で参考になる点**
    | 5位のやり方 | 私たちへの落とし込み |
    |-------------|----------------------|
    | **ProNE（グラフ埋め込み）** | user–anime の「視聴」二部グラフで ProNE → user ベクトル・anime ベクトル（128 次元）と類似度。同様に user–genre, user–producer, user–studio。 | **批評家–映画**の「レビューした」二部グラフで ProNE（または Node2Vec 等）→ 批評家ベクトル・映画ベクトルと類似度。既存の NMF/SVD と別軸の「グラフ埋め込み」になる。 |
    | **KMeans でクラスタ距離** | 埋め込みを KMeans(100) でクラスタリングし、**各サンプルからクラスタ中心までの距離**を特徴量。 | 批評家・映画の埋め込みをそれぞれクラスタリングし、距離列を足す。 |
    | **スコア重み付き埋め込み** | user–anime グラフで、user の隣接 anime ベクトルを **score の昇順 pct_rank²** と **降順 pct_rank²** で重み付き平均 → 2 種類の user ベクトル。 | 批評家の「隣接映画」ベクトルを **Fresh 率が低い順（Rotten 寄り）の重み** と **高い順（Fresh 寄り）の重み** でそれぞれ加重平均 → 批評家あたり 2 本のベクトル。 |
    | **スコア考慮グラフで ProNE 再計算** | score の昇順・降順 pct_rank×10 だけ仮想的に anime ノードを増やしたグラフで ProNE → 256 次元 user ベクトル（低スコア寄り・高スコア寄り）。 | Fresh/Rotten を 0/1 とし、「Rotten 寄り」「Fresh 寄り」でエッジを重み付けまたは仮想ノード増やしてグラフを変え、再度埋め込み。実装コストは高め。 |
  - **効かなかったこと（5位）**: Unseen 側で各 fold をさらに分割して OOF 特徴量にすると悪化。スコア考慮 ProNE の一貫性が崩れた可能性。
  - **私たちですでにやっていること**: 既知/未知二モデル、GroupKFold（映画）。**まだ**: グラフ埋め込み（ProNE 等）、スコア重み付き隣接ベクトル、CV スコアの加重（0.7×時系列 + 0.3×GroupKFold）は試していない。

- **(I) atmaCup #15 2位解法（ynktk）— 特徴の多様化・スタッキング・特徴選択**
  - **スコア**: CV 1.085（Ridge スタッキング後）, Public 1.1740, Private 1.1361。同一 2362 次元特徴を LightGBM / XGBoost / CatBoost で学習し、Ridge でメタ学習。
  - **Validation**: KFold(10, shuffle=True)。時系列・GroupKFold は使っていないが、特徴量は train+test 結合で事前作成。
  - **特徴量で参考になる点**
    | 2位のやり方 | 私たちへの落とし込み |
    |-------------|----------------------|
    | **数値の四則演算** | 数値特徴の組み合わせで和・差・積・商を追加。 | movie_age_days, runtime, review_year などの 2 列組み合わせで 1〜2 種類だけ試す。 |
    | **カテゴリの Word2Vec** | 全カテゴリを行方向に結合して 1 文とみなし、Word2Vec で学習。 | critic_name, publisher_name, movie_title などを結合した「文」を Word2Vec（または doc2vec）→ 批評家・映画の埋め込み。既存の SVD と別軸。 |
    | **マルチラベル → SVD** | MultiLabelBinarizer → SVD で次元削減。マルチラベル全体を結合して Word2Vec も。 | genres の one-hot → TruncatedSVD。既存の genre  one-hot に加えて SVD 圧縮列を足す。 |
    | **テキスト TF-IDF + SVD** | japanese_name の TF-IDF → SVD。文字数・単語数も。 | movie_info / movie_title の TF-IDF → SVD は 01 で実施済み。2 位は「名前」ベースなので私たちはテキスト系で近い。 |
    | **id2vec の window・次元** | user×anime の「視聴シーケンス」を document にし、window=400, embedding=256。**window は大きめ、embedding 次元の寄与も大きい**。seed が固定されず「いい特徴を引くまでガチャ」。 | 批評家×映画を review_date でソートし「批評家の映画シーケンス」を document に（既存の SVD と同様）。**window を 200〜500 程度に**、**n_components を 64〜256 で試す**。複数 seed で取り直してアンサンブル候補にする。 |
    | **複数の Vectorizer + 分解** | (CountVectorizer, LDA), (CountVectorizer, SVD), (CountVectorizer, NMF), (TfidfVectorizer, LDA) を user×anime, user×original_work_name で実施。n_components=100。 | 批評家の「レビューした映画 ID 列」を document に、CountVec+SVD / CountVec+NMF / Tfidf+LDA など**複数組み合わせ**で特徴量を作る。既存は CountVec+SVD のみなので NMF・LDA を足す。 |
  - **特徴選択**: CatBoost で各 CV fold の重要度を計算し、**全 fold で重要度 0 の特徴だけ除去**。私たちは未実施なので、特徴が増えすぎたときに 1 本入れるとよい。
  - **モデル・スタッキング**: 同じ特徴で LightGBM / XGBoost / CatBoost を学習し、**Ridge でスタッキング**（メタ特徴は各モデルの予測値）。CV で 1.12→1.09→1.085 と段階的に改善。私たちは LightGBM のみなので、**XGBoost・CatBoost を足して Ridge メタ**がそのまま使える。
  - **私たちですでにやっていること**: テキスト TF-IDF+SVD、批評家×映画の CountVec+SVD（2位の anime2vec に相当）。**まだ**: 数値の四則演算列、カテゴリ Word2Vec、マルチラベル SVD、複数 Vectorizer+分解（NMF/LDA）、特徴選択、多モデル+Ridge スタッキング。

→ まとめると、**「映画単位のテキスト埋め込み + GBDT」** という今の軸は、類似タスクの上位解法と方向性が合っている。  
残り時間とコストを見ながら、

1. OpenAI embedding small / large や次元数のパターンを増やしてシードアベレージ/アンサンブル  
2. 余裕があれば TF-IDF-SVD や BERT 埋め込みを「もう1本のビュー」として追加

という順で試すのが現実的なロードマップ。

---

### 7.2 上位解法から試すときのロジック・手順（実装でずれないように）

以下は (F)(G)(H)(I) から「試せそう」なものを、**ロジックがずれないよう**手順レベルで書いたもの。実装時はこの順序・条件を守ること。

---

#### (1) 類似度を使った TE（6位・Implicit の Jaccard）

**目的**: 「似た批評家だけ」「似た映画だけ」で TE を計算し、ノイズを減らす。

**前提**: 予測対象は「行」(批評家, 映画) の組。その行の target は特徴量・類似度計算に**一切使わない**（リーク防止）。

**手順（批評家側 TE「似た批評家 top-k の、その映画の Fresh 率」）**

1. **批評家ごとの「レビューした映画の集合」を用意する**
   - 時系列を守る場合: **その行の review_date より前**のレビューのみで集合を作る。つまり行 i に対しては、`train[train.review_date < 行iのreview_date]` で批評家別に `rotten_tomatoes_link` を集め、`critic → set(movie_id)` の辞書を作る（行 i の批評家の「その行より前」の集合には、**行 i の映画は含めない**）。
   - 時系列を無視する場合: 全 train で批評家ごとに映画集合を作るが、**TE の平均を取るときに「予測対象の (critic, movie) のラベルは除く」**必要がある（6位の「寄与を引く」に相当）。

2. **批評家間の類似度（Jaccard）を定義する**
   - 批評家 A, B の集合を \( S_A, S_B \) とする。  
     \( J(A,B) = (|S_A \cap S_B| + 1) / (|S_A \cup S_B| + 1) \)（0 除算・空対策で +1）。  
   - 行 i の批評家を \( c_i \) とする。**全批評家** \( v \) に対して \( J(c_i, v) \) を計算し、**\( c_i \) 自身を除いた**うえで降順にソートし、top-k の批評家リスト \( N_k(c_i) \) を得る。

3. **「その映画」に対する top-k 批評家の Fresh 率を計算する**
   - 行 i の映画を \( m_i \) とする。
   - **train のうち「批評家が \( N_k(c_i) \) のいずれか かつ 映画が \( m_i \)」の行**だけを抜き出す。その行の `target` の平均を TE 値とする。
   - **行 i 自身はこの平均に含めない**（行 i の批評家は \( N_k(c_i) \) に含まれうるが、行 i の (critic, movie) のラベルは使わない）。  
   - 該当行が 0 件のときは global mean や全体の Fresh 率で埋める。

4. **映画側 TE「似た映画 top-k の、その批評家の Fresh 率」**
   - 映画ごとの「レビューした批評家の集合」を同様に用意（時系列ならその行より前のみ、かつその行の批評家は含めない）。
   - 映画間 Jaccard を計算し、行 i の映画に似た映画 top-k を取得。
   - train のうち「映画がその top-k のいずれか かつ 批評家が行 i の批評家」の行の target の平均を計算（行 i 自身は除く）。

**k のバリエーション**: k = [1, 3, 5, 10, 50] など、各 k で上記 2 本（批評家側 TE・映画側 TE）を特徴量にする。同じ類似度で複数 k を出すと 2×len(k) 列になる。

---

#### (2) NMF / embedding の次元数バリエーション + 全データ 1 本のアンサンブル（3位）

**目的**: 次元数違いで多様な予測を取り、さらに「全 train で 1 本学習したモデル」を足してアンサンブルする。

**手順**

1. **特徴量は同一**（例: ベースライン38 + embedding の PCA）。**PCA の n_components だけ**を 8, 16, 32 などに変えた複数セットを用意する（各セットで別々に学習するので、特徴量名が被らないよう prefix を変えるか、別ノート/別実行にする）。
2. **時系列 CV** で各設定を学習し、fold ごとに test を予測。各設定で **fold 予測の平均** を 1 本の予測とする。
3. **全 train で 1 本**、同じ特徴量で学習し、test を予測する（early_stopping は使わないか、valid を 1 割 holdout にするなど）。
4. **アンサンブル**: 上記「時系列 CV の fold 平均」複数設定 + 「全データ 1 本」の予測を、**均等平均または CV に合わせた重み**で平均する。3 位は「全データ 1 本を足すと Public が上がった」としているので、少なくとも「CV 各 fold の平均」と「全データ 1 本」の 2 本を平均するのを試す。

**ロジックの要点**: 全データ 1 本は **検証データを一切使っていない** ので、CV スコアでは評価できない。提出用の「ブレンド用の 1 本」として扱う。

---

#### (3) CV スコアの加重（5位）

**目的**: テストで未知映画が一定割合あるとき、時系列 CV（既知寄り）と GroupKFold（未知寄り）を**比率に合わせて 1 本の CV 指標**にする。

**手順**

1. テストにおける**未知映画の割合**を推定する（学習に登場する `rotten_tomatoes_link` と test のそれを比べ、test で新規の映画の割合を p とする）。不明なら 0.2〜0.3 程度を仮定。
2. 時系列 CV の AUC を \( A_{ts} \)、GroupKFold（group=映画）の AUC を \( A_{gk} \) とする。
3. **加重 CV スコア** = \( (1-p) \times A_{ts} + p \times A_{gk} \) をモデル比較用の 1 本の指標にする。  
   - 採用判定は「加重 CV がベースより高いか」で行う。提出は従来どおり時系列 or 既知/未知二モデルでよい。

**ロジックの要点**: 時系列と GroupKFold は**別の分割**なので、両方の AUC を出したうえで線形結合するだけ。学習手順は変えない。

---

#### (4) 特徴選択（2位・全 fold で重要度 0 の除去）

**目的**: ノイズ列を除いて過学習を抑える。

**手順**

1. **同じ CV 分割**（例: 時系列 4 fold）で、**各 fold ごとに** LightGBM または CatBoost を学習する。
2. 各 fold で **feature_importances_**（または gain）を取得し、特徴名と重要度を記録する。
3. **全 fold で重要度が 0 だった特徴**の名前のリストを作る。
4. そのリストに含まれる列を、**学習・予測の特徴量から除外**する。除外した特徴量だけで再度 CV を回し、CV スコアが維持または改善するか確認する。

**ロジックの要点**: 「全 fold で 0」でないと、たまたま 1 fold でだけ使われていない列を消してしまう可能性がある。**重要度が 0 でない列は残す**。

---

#### (5) 多モデル + Ridge スタッキング（2位）

**目的**: LightGBM 単体より、LGB・XGB・CatBoost の予測を Ridge でまとめたほうが CV が伸びることがある。

**手順**

1. **同一の特徴量**で、同じ CV 分割（例: 時系列 4 fold）を用意する。
2. **各 fold で**:
   - train/val で LightGBM, XGBoost, CatBoost をそれぞれ学習する（early_stopping は val で共通）。
   - val に対して 3 モデルの予測 \( \hat{p}_{LGB}, \hat{p}_{XGB}, \hat{p}_{Cat} \) を出す。これが **メタ特徴量**（3 次元）。
   - **Ridge 回帰**（または LogisticRegression）を、メタ特徴量 (3 列) を入力・val の正解 target を目的変数として学習する。正則化は CV で決める。
3. **OOF と test 予測**:
   - 各 fold の val に対して、その fold で学習した Ridge が 3 モデル予測から出す値を OOF とする（4 fold 分を結合して OOF が 1 本できる）。
   - test については、**各 fold の 3 モデルで test を予測**し、各 fold の Ridge でその 3 予測をまとめる。最後に **fold ごとのメタ予測の平均** を test の最終予測にする。
4. **CV スコア**: OOF と正解の AUC を計算する。この OOF は「Ridge を通したあと」の予測なので、単体モデルより改善しているかを見る。

**ロジックの要点**: メタモデル（Ridge）の学習には **val の正解**を使う。test は「各 fold の 3 モデル予測 → その fold の Ridge で 1 本に → fold 平均」であり、**test の正解は使わない**。Ridge の入力は「3 モデルの予測値」のみで、元特徴量は使わない（スタッキングの典型形）。

---

#### (6) スコア重み付き隣接ベクトル（5位）

**目的**: 批評家の「隣接映画」の埋め込みを、「Rotten 寄り」と「Fresh 寄り」で重みを変えて 2 種類の批評家ベクトルを作る。

**手順**

1. **批評家–映画の埋め込み**を既に持っているとする（例: NMF や ProNE で批評家ベクトル \( u \)、映画ベクトル \( m \) が行ごとにある）。
2. 行 i（批評家 \( c_i \)、映画 \( m_i \)）について、**その批評家がレビューした映画**のリストと、そのときの **target（0/1）** を取得する。**行 i 自身は除く**（リーク防止）。時系列なら「行 i の review_date より前」のレビューのみ。
3. その批評家の「隣接映画」のベクトルを、**target の昇順（Rotten が上）の pct_rank を²したもの**で重み付き平均 → ベクトル \( v_{low} \)。同様に **target の降順（Fresh が上）の pct_rank²** で重み付き平均 → ベクトル \( v_{high} \)。
4. \( v_{low}, v_{high }\) をその行の特徴量として追加する（次元は埋め込み次元と同じ。2 本で 2×dim 列）。
5. **学習時**: 上記は **target を使っている**ので、**時系列 CV の各 fold では「その fold の train だけ」で重みと平均を計算**し、val/test にはその重み・中心で変換する（val の行に対しては、train 側の「批評家ごとの隣接映画と target」だけを使って \( v_{low}, v_{high} \) を計算する）。映画が train にない場合は、その映画ベクトルは「隣接」に含めず、重み付き平均は「train に存在する隣接のみ」で行う。

**ロジックの要点**: 重み付き平均の**重みに target を使う**ため、**fold の train だけで重みと平均を決め、val/test には適用するだけ**にしないとリークする。映画側についても「その映画にレビューした批評家」で同様に 2 本作る場合は、同じく fold の train 内だけで計算する。

---

#### (7) 複数 Vectorizer + 分解（2位・批評家の映画 ID 列）

**目的**: 批評家の「レビューした映画 ID の列」を document とみなし、CountVec+SVD に加えて CountVec+NMF, Tfidf+LDA など複数パターンで特徴量を作る。

**手順**

1. **document の作り方**: 批評家ごとに、**review_date でソートした**「レビューした映画 ID」（`rotten_tomatoes_link`）の列をスペース区切りで 1 文にする。時系列を守る場合は、**予測対象の行の review_date より前**のレビューのみで document を作る（fold ごとに train だけを使う）。
2. **CountVectorizer**: 語彙は「映画 ID」の文字列。`token_pattern` はデフォルトだと 2 文字以上になるので、映画 ID が 1 文字のときは `token_pattern=r'\S+'` などでトークン化。`max_features` は映画数程度。
3. **TfidfVectorizer**: 同様に映画 ID をトークンに。Count と Tfidf で分布が変わる。
4. **分解**: Count または Tfidf の行列に対して、**TruncatedSVD**, **NMF**, **LDA** のいずれかを fit。`n_components` は 32〜100 程度。**fit は fold の train のみ**で行い、val/test は `transform` だけ（時系列 CV なら各 fold の train で fit）。
5. **行への付与**: 各行は (critic_id, movie_id) に対応。批評家側の埋め込みは、その批評家の document から得たベクトル（1 本）。映画側は、その映画が「どの批評家の document に出現するか」から逆に映画ベクトルを作る（2 位の anime2vec の「user の document → user ベクトル」に加え「anime の document」も作る場合）。私たちは既存で批評家の document → SVD があるので、**同じ document で NMF, LDA を追加**し、列名を `nmf_c_0` などと分ける。

**ロジックの要点**: **fit は必ず「その fold の train だけ」**。val/test は train で fit した Vectorizer と分解モデルで `transform` のみ。映画 ID が train にしかない場合、test の批評家の document には「未知の映画 ID」が含まれるが、Vectorizer は train の語彙で学習しているので、未知 ID は無視される（または OOV として 0 になる）。そのままでよい。

---

#### (8) 未知用モデルで「未知映画の行は TE に使わない」（5位）

**目的**: 未知映画用モデルでは、未知映画の行の target を特徴量計算（TE 等）に含めない。

**手順**

1. **GroupKFold（group=映画）**で train を分割する。fold の「val」= 検証に出す映画の行全体。
2. **未知用の 1 fold では**、学習データ = その fold の train の行。ただし **「train に含まれる映画」のうち、別の fold の val に含まれる映画**（＝検証側の映画）は、**この fold の「特徴量計算用のデータ」からは除外する**必要はない（GroupKFold の train には検証映画の行は元々含まれていない）。
3. 5 位でやっているのは、**unseen user 用の train** を「seen_train / unseen_train」にさらに分割し、**unseen_train の score を NaN にして**特徴量計算に使わない、というもの。私たちでは「未知映画の行」が、学習データに含まれるかどうかがポイント。
4. **私たちの既知/未知二モデル**では、未知用は「映画が train に登場しない」行の予測に使う。未知用モデルの**学習**には、train の全行を使うが、**TE は「映画 ID ごとの Fresh 率」などを使うと、未知映画は train に 1 回も出てこないので TE が未定義になる**。そこで未知用では、**映画 ID をキーにした TE は使わない**か、**train に登場する映画だけ**で TE を計算し、未知映画には global mean を渡す、という設計にする。5 位の「unseen の score を NaN にして特徴量に使わない」は、**未知ユーザーの行のラベルを TE の集計に含めない**という意味なので、私たちでは「未知映画の行の target を、映画別 TE の計算に含めない」に対応する。実装では、**映画 ID ごとの TE を計算するときに、その映画が「未知用 fold の val に含まれる映画」なら、その映画の行の target を集計から除く**のではなく、**未知用モデルでは映画 ID の TE 自体をやめて、批評家・publisher など「映画に依存しない」特徴だけにする**か、または **映画をクラスタや埋め込みで表現し、そのクラスタ/埋め込みの TE だけ使う**などの方針にする。

**ロジックの要点**: 未知映画は学習データに**映画として**登場しないので、「映画 ID をキーにした TE」は未知映画に対しては定義できない。だから未知用モデルでは、映画 ID TE を使わないか、映画を別の離散化（クラスタなど）にマップしてから TE を取る、という設計にする。

---

以上を守れば、上位解法のロジックを崩さずに試せる。追加で「Explicit 類似度（6位）」「ProNE グラフ埋め込み（5位）」を試す場合は、リーク防止（時系列 or fold 内 train のみで類似度・重みを計算）と、予測行の (critic, movie) を平均から除外することを同じように細かく決めてから実装すること。

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
