# 協調フィルタリング系の改善案と実装

atmaCup #15 1位解法を参考にした**協調フィルタリング・接続情報活用**の改善案と、現状できている実装をまとめる。  
当コンペでは **批評家 = user、映画 = item、「レビューした」= 接続** に対応づける。

---

## 0. 協調フィルタリングの計算の流れ（噛み砕き）

「協調フィルタで何を計算しているか」を、入力から最終的な特徴まで順にまとめる。

### 0.1 入力になるもの

- **使う情報**: 「**誰がどの映画をレビューしたか**」だけ。  
  Fresh か Rotten かは使わない（接続の有無だけ = implicit feedback）。
- **単位**: 1 行 = 1 件のレビュー（批評家 A が映画 M をレビューした、という 1 件）。

### 0.2 行列にする

- train と test の**全レビュー**について、「批評家 ID」と「映画 ID」のペアを集める。
- **批評家 × 映画** の大きな表（行列）を作り、  
  「その批評家がその映画をレビューしていれば **1**、していなければ **0**」を入れる。  
  （実際のコードでは 1 のところだけを記録する疎な行列にしている。）
- この行列が「**誰が何をレビューしたか**」の 0/1 の地図になる。

### 0.3 行列分解（BPR または ALS）でベクトルを作る

- 上の 0/1 行列を **BPR** や **ALS** というアルゴリズムで「分解」する。
- 分解の結果、次の 2 つが得られる：
  - **批評家ベクトル**: 批評家ごとに、長さ 16（または 32 など）の数値のベクトル。
  - **映画ベクトル**: 映画ごとに、同じ長さの数値のベクトル。
- イメージとしては、  
  「**似たレビュー履歴の批評家は似たベクトル**」「**似た批評家にレビューされる映画は似たベクトル**」になるように中身が決まる。
- このベクトルたちが「**協調フィルタの計算結果**」の中身。

### 0.4 各行の特徴としてどう使うか

- 予測したいのは「**この 1 件のレビュー（批評家 A × 映画 M）が Fresh か Rotten か**」。
- そこで、**その行に対応する批評家のベクトル** と **その映画のベクトル** を、その行の特徴として使う。
- **パターン A（今のメイン）**:  
  批評家ベクトル 16 個の数値 + 映画ベクトル 16 個の数値 = **32 列** を、もともとの 55 特徴に**足して**、LightGBM に渡す。  
  → 合計 55 + 32 = 87 特徴で「この組み合わせの傾向」を学習させる。
- **パターン B（内積 1 列）**:  
  批評家ベクトルと映画ベクトルを**内積**（対応する成分のかけ算の和）して **1 つの数** にする。  
  その 1 列だけを 55 特徴に足す。  
  → 「この批評家×この映画の相性スコア」を 1 つの特徴として渡す。

### 0.5 2-hop 集約とは（別の計算）

- 協調フィルタの「行列分解」とは別に、**2-hop** は次のような計算。
- **その映画をレビューした批評家たち**を、train のなかで集める。
- その批評家たちについて、  
  「Fresh になった割合の平均」「critic_te の平均」「レビュー数」などを計算する。
- その 1 つずつを**新しい 1 列**として、その行（その批評家×その映画）に付ける。
- 意味としては「**この映画をレビューした人たちは、全体として Fresh にしやすいか / どんな傾向か**」を数値にしたもの。
- **2-hop の割り算**:  
  「この批評家の critic_te」÷「その映画をレビューした批評家の critic_te の平均」を 1 列にする。  
  1 位で効いた「値 ÷ ユーザー平均」の、映画側バージョン。

### 0.6 まとめ（何がベースで何を足しているか）

- **現在の最高スコア 0.76101** は、  
  **55 特徴（embedding + PCA16 + doc_x_critic_te）** に、  
  **BPR で作った批評家・映画ベクトル 32 列** を足した **87 特徴** で学習した 1 本。
- 2-hop 実験ノート（`train_2hop_experiments.ipynb`）の **ベースはこの 1 本と同じ**。  
  `get_setup(use_best_pipeline=True)` で 55 特徴を用意し、`get_bpr_base(ctx, factors=16)` で同じ BPR 16 の 32 列を足している。  
  したがって「BPR のみ」の実験（`run_experiment(ctx, "bpr_only")`）は、0.76101 を出した提出と**同じ特徴・同じベース**になっている。

---

## 1. 改善案一覧

| # | 案 | 1位の要点 | 当コンペでの具体 | 実装状況 |
|---|-----|-----------|------------------|----------|
| **1** | 批評家ごと集約 + 割り算 | 値 / ユーザーごとの集約(平均) が特に効いた。好みからどれだけ離れているかを表す。 | 批評家ごとに train 内で数値特徴の平均を計算し、**現在行の値 / その批評家の平均** を 1 列追加。未知批評家は global 平均で割る。 | ✅ 実装済み |
| **2** | Implicit 埋め込み | train+test の (user, item) を結合し、implicit で埋め込みを得て特徴量に。接続の有無だけ（評価値は使わない）。 | (critic, movie) を train+test で結合 → ALS/BPR で批評家・映画ベクトルを学習 → 各行に (critic_vec \| movie_vec) を追加。 | ✅ 実装済み |
| **3** | Seen / Unseen 二モデル | seen 用と unseen 用で別モデル。CV は seen×0.77 + unseen×0.23。 | 既知批評家用・未知批評家用の 2 モデルを作り、test 行ごとにどちらかの予測を使う。ブレンド重みを変えて複数提出。 | **当コンペでは不要**（EDA で test の批評家は 100% train に登場・unseen=0。§6 参照） |
| **4** | 2-hop 集約 | そのアイテムを視聴したユーザーたちの集約をさらに集約して 2-hop の特徴に。 | 映画 m について「m をレビューした批評家たち」の Fresh 率平均・critic_te 平均・人数などを 1 列ずつ追加。 | ✅ `lib/two_hop.py`（BPR ベースに追加して実験用） |
| **5** | 割り算 + 2-hop の組み合わせ | 集約と割り算の組み合わせが効く。 | 割り算 1 列と 2-hop 1 列を組み合わせたバリエーションで提出。 | 未実装 |

---

## 2. 現状の実装

### 2.1 コード（lib/improvement_candidates.py）

- **`run_atmacup_ratio(ctx, base_feature, suffix)`**  
  - 批評家ごとに train のみで `base_feature` の平均を計算。  
  - 特徴量「値 / 批評家平均」を 1 列追加（0 除算は 1.0、未知批評家は global 平均）。  
  - 55 特徴 + 1 列で LGB 学習し、`submission_atmacup_ratio_{suffix}.csv` を保存。

- **`run_atmacup_implicit(ctx, method, factors, suffix)`**  
  - train+test の (critic, movie) を 0/1 の交互行列にし、`implicit` の ALS または BPR で fit。  
  - `user_factors`（批評家ベクトル）と `item_factors`（映画ベクトル）を各行に付与し、55 特徴に 2×factors 列を追加。  
  - LGB 学習し、`submission_atmacup_implicit_{suffix}.csv` を保存。

### 2.2 実行ノートブック（train_atmacup_5patterns.ipynb）

**ベース**: 0.75493 の土台（55 特徴: embedding + PCA16 + doc_x_critic_te）。この 5 パターンから **implicit BPR 16 で 0.76101**（現時点の最高）を達成。

**5 パターン**（提出ファイル）:

| # | 内容 | 提出ファイル名 |
|---|------|----------------|
| 1 | 割り算: genre_Documentary / 批評家平均 | submission_atmacup_ratio_genre_doc.csv |
| 2 | 割り算: critic_name_te_ts / 批評家平均 | submission_atmacup_ratio_critic_te.csv |
| 3 | Implicit ALS factors=16 | submission_atmacup_implicit_als16.csv |
| 4 | Implicit BPR factors=16 | submission_atmacup_implicit_bpr16.csv |
| 5 | 割り算: runtime_bin / 批評家平均 | submission_atmacup_ratio_runtime_bin.csv |

**依存**: `pip install implicit`（requirements.txt に追加済み）。

### 2.3 BPR ベース + 2-hop 実験（lib/two_hop.py / train_2hop_experiments.ipynb）

- **ベース**: **0.76101 を出した提出と同一**。`get_setup(use_best_pipeline=True)` で 55 特徴、`get_bpr_base(ctx, factors=16)` で BPR 16 の 32 列を足す。これは `run_atmacup_implicit(ctx, "bpr", 16, "bpr16")` が内部で使っているのと同じ `_build_implicit_embeddings` を `get_bpr_base` が呼ぶため、同じ 87 特徴になる。したがって「BPR のみ」の実験は最高スコアの土台そのもの。
- **`lib/two_hop.py`**  
  - `add_2hop_features(train_df, test_df, columns)` … 映画ごとに train で集約（Fresh 率平均・critic_te 平均・レビュー数）を計算し、指定列だけ train/test に追加。  
  - `run_experiment(ctx, experiment_name, use_2hop_cols=None)` … BPR ベースで学習し、`use_2hop_cols` で指定した 2-hop 列を追加して `submission_2hop_{experiment_name}.csv` を保存。  
- **ノートブック**: `train_2hop_experiments.ipynb` で「BPR のみ」「BPR + 2-hop 1 列」「BPR + 全2-hop」などを 1 本ずつ実行し、Public スコアの増減を見ながら追加する列を決められる。

---

## 3. 1位解法との対応検証

### 3.1 割り算（run_atmacup_ratio）

| 1位の要件 | 実装 | 一致 |
|-----------|------|------|
| 集約前の値 / ユーザーごとの集約結果(平均) | 現在行の `base_feature` / 批評家ごとの `base_feature` の平均（train のみ） | ✅ |
| そのアイテムが好みからどれだけ離れているか | 比が 1 に近い＝傾向どおり、1 から離れる＝傾向から外れている | ✅ |
| 分母はユーザー(批評家)ごとの集約 | `groupby("critic_name")[base_feature].mean()` | ✅ |
| 未知ユーザー | `fillna(global_mean)` で分母を global 平均に | ✅ |
| 0 除算 | 分母 0 を `replace(0, np.nan).fillna(1.0)` で 1.0 に | ✅ |

**補足**: 批評家平均は train 全体で計算（時系列で「その行より前」だけにはしていない簡易版）。test は平均に使わないので test へのリークはない。

### 3.2 Implicit 埋め込み（run_atmacup_implicit）

| 1位の要件 | 実装 | 一致 |
|-----------|------|------|
| train, test の (user_id, item_id) ペアを結合 | train と test の (critic, movie) を両方行列に追加 | ✅ |
| 接続の有無だけ（評価値は使わない） | 行列の値はすべて 1。Fresh/Rotten は使っていない | ✅ |
| implicit で埋め込みを得る | ALS / BPR で fit、user_factors / item_factors を取得 | ✅ |
| 埋め込みを特徴量に | 批評家ベクトル + 映画ベクトルを 55 特徴に追加 | ✅ |
| 行列の形 | (users, items) = (n_critics, n_movies) の CSR。implicit の規約に準拠 | ✅ |

**協調フィルタリングの考え方（実装の流れ）**  
1. 批評家×映画の「誰がどの映画をレビューしたか」を 0/1 行列にする。  
2. ALS/BPR で低ランク分解し、批評家ベクトル・映画ベクトルを得る（似た批評家＝似たベクトル、似た映画＝似たベクトル）。  
3. 予測は内積で行わず、そのベクトルを**特徴量**として LGB に渡し、Fresh/Rotten を予測する。

---

## 4. 拡張の余地

| 軸 | 現状 | 拡張例 |
|----|------|--------|
| **割り算** | 1 特徴 1 列・批評家平均のみ | 複数特徴を同時に足す、**時系列で「その行より前」だけ**で批評家平均を取る、映画側平均で割る |
| **Implicit** | ALS/BPR・factors=16・ベクトル結合 | factors=8/32、**内積 1 列**のみ、confidence 重み、他手法 |
| **2-hop** | 未実装 | 「その映画をレビューした批評家」の統計を 1 列ずつ追加 |
| **Seen/Unseen** | 未実装 | 既知/未知の 2 モデル＋ブレンド |
| **組み合わせ** | 割り算と implicit は別提出 | 55＋割り算＋implicit を 1 本に載せる、スタッキングのメタ特徴に使う |

実装は `run_atmacup_ratio` / `run_atmacup_implicit` を土台に、パラメータ追加や別関数の追加で上記を載せていける形になっている。

### 4.1 行列分解の別手法（ALS/BPR 以外）

| 手法 | 考え方 | 備考 |
|------|--------|------|
| **ALS** | 「この批評家×この映画は 1 に近い／0 に近い」を**値**として当てる。二乗誤差を交互に最小化。 | ✅ 実装済み。0.75556。 |
| **BPR** | 「この批評家には、レビューした映画の方がしてない映画より上に来る」**順序**を当てる。ランキング学習。 | ✅ 実装済み。0.76101（現最高）。 |
| **NMF** | 非負制約で分解。値の**非負の** latent の積で表現。ALS と異なる構造を捉える可能性。 | 未実装。`sklearn.decomposition.NMF` で同じ 0/1 行列を分解し、W (n_critics × k), H (k × n_movies) から批評家・映画ベクトルを得る。 |
| **TruncatedSVD** | 0/1 行列をそのまま SVD。特異値分解で低ランク近似。値の再構成誤差を最小化。 | 未実装。`sklearn.decomposition.TruncatedSVD` で同じ行列を分解。U が批評家、V が映画側の表現。 |
| **Logistic MF** | 相互作用を**確率**でモデル化（ロジスティック関数）。implicit 向けの論文あり。 | 未実装。`implicit` には標準でない。別ライブラリ（logistic-mf 等）か自前で試す候補。 |
| **内積 1 列** | 手法は ALS/BPR のまま。埋め込み 32 列ではなく **user_vec · item_vec の 1 列だけ**を特徴に追加。 | 未実装。CF の「この組み合わせのスコア」を直接 LGB に渡す。 |

### 4.2 明日試すべき実装（優先度）

以下を**明日試す**とよさそう。既存の `run_atmacup_implicit` やパイプラインを少し変えるだけで試せるものから並べる。

| 優先度 | 内容 | やること |
|--------|------|----------|
| **1** | **BPR の factors を増やす** | `run_atmacup_implicit(ctx, "bpr", 32, "bpr32")`、同様に 64。現状 16 で 0.76101 なので、次元を増やして効くか確認。 |
| **2** | **内積 1 列** | ALS または BPR で得た `user_factors`, `item_factors` を使い、各行で `score = user_factors[critic_id] @ item_factors[movie_id]` を 1 列だけ追加。55＋1 で LGB。CF の予測を直接特徴にする。 |
| **3** | **NMF 埋め込み** | 同じ (critic, movie) 0/1 行列を `sklearn.decomposition.NMF(n_components=16)` で分解。`model.transform(行列.T)` で映画側、`W` で批評家側のベクトルを取り、ALS/BPR と同様に 55＋32 列で LGB。 |
| **4** | **2-hop 集約** | 映画 m について「m をレビューした批評家たち」の train 内での Fresh 率平均・人数などを 1 列ずつ追加。既存の改善案 §1 の #4。 |
| **5** | **ALS の factors 増加** | `run_atmacup_implicit(ctx, "als", 32, "als32")`。BPR が効いているので ALS も次元を上げて試す。 |
| **6** | **ALS/BPR のハイパラ** | `regularization`, `iterations` などを変えて CV が良くなる設定を探す。 |

- **実装の置き場所**: 1・5 は既存ノートで `run_atmacup_implicit` の第 3 引数を変えるだけ。2 は `lib/improvement_candidates.py` に `run_atmacup_implicit_dot(ctx, method, factors, suffix)` のような関数を追加。3 は NMF 用の `run_atmacup_nmf` を追加。4 は 2-hop 用の特徴量生成を pipeline か improvement_candidates に追加。

---

## 5. スコア上昇は期待できるか

**結論（結果反映）**: **implicit 埋め込み（BPR/ALS）は大きく効いた**（BPR で 0.76101、従来ベースライン 0.75493 を更新）。**割り算**は単体ではベースライン前後〜未満なので、伸ばすなら implicit を軸にするのが有効。

- **足しすぎで落ちた実績**: 55 特徴ベースに 1 列足しただけ（year_norm_x_critic_te）で 0.75361、TF-IDF で 0.75386、特徴選択で 0.75450 など、**列追加だけでは 0.75493 を超えにくい**。過学習・分布ずれで Public が下がりやすい傾向がある。
- **implicit は別**: 接続の有無を行列分解して埋め込みとして足すと、当コンペでは BPR 0.76101・ALS 0.75556 と効いている（§5 下の表参照）。
- **割り算**: 批評家平均で割る 1 列は単体では 0.753〜0.754 台。1位解法では効いたが、当コンペでは implicit ほどは伸びていない。

**試す価値**: 1 本ずつ提出して Public を確認するのは意味がある。伸びなければ「この土台では足し算は効きにくい」という知見になる。伸びればラッキー、というスタンスが現実的。

**Public 結果（協調フィルタ系 5 本）**:

| 提出 | Public Score | 備考 |
|------|--------------|------|
| submission_atmacup_implicit_bpr16.csv | **0.76101** | **現時点の最高。** 協調フィルタ（implicit BPR 埋め込み）が大きく効いた。 |
| submission_atmacup_implicit_als16.csv | 0.75556 | 旧ベースライン 0.75493 を上回る。ALS も有効。 |
| submission_atmacup_ratio_runtime_bin.csv | 0.75457 | 割り算は微増〜ベースライン前後。 |
| submission_atmacup_ratio_critic_te.csv | 0.75364 | 割り算単体では伸びず。 |
| submission_atmacup_ratio_genre_doc.csv | 0.75360 | 同上。 |

- **ベースライン更新**: 従来の最高は 0.75493（55 特徴 doc_x_critic_te）。**implicit BPR 16 で 0.76101 を達成し、協調フィルタが本コンペでかなり効くことが分かった。** 今後の改善は「0.76101 を超える」を目標にするとよい。
- **結論**: 協調フィルタ（とくに implicit 埋め込み）がゲー。割り算は単体では弱いが、BPR/ALS の特徴追加は強く効いている。

---

## 6. Seen/Unseen（批評家基準）の実装案

1位解法では **user（＝当コンペでは批評家）** で「train に登場した user = seen」「登場していない = unseen」とし、seen 用・unseen 用の 2 モデルを組み合わせている。当コンペでは **批評家基準** で同じことをする。

### 6.1 EDA（seen/unseen の割合）

- **スクリプト**: `scripts/eda_seen_unseen_critics.py`  
  - train/test の `critic_name` のみを読み（train はチャンク読みでメモリ節約）、次を集計する。  
  - 集計内容: train の行数・批評家ユニーク数、test の行数・批評家ユニーク数、**test の批評家のうち train に登場する数（seen）・登場しない数（unseen）**、**test の行数ベースで seen 批評家の行数・unseen 批評家の行数**。  
- **注意**: train.csv が約 730MB あるため、フル実行すると数分かかることがある。

**実行結果（当コンペ）**:

| 項目 | 値 |
|------|-----|
| Train | 653,507 行・1,573 ユニーク批評家 |
| Test | 40,716 行・1,483 ユニーク批評家 |
| Test の批評家のうち train に登場（seen） | 1,483（**100%**） |
| Test の批評家のうち train に未登場（unseen） | **0（0%）** |
| Test 行のうち seen 批評家 | 40,716（100%） |
| Test 行のうち unseen 批評家 | 0（0%） |

**結論**: 当コンペでは **test の全批評家が train に登場しており、unseen 批評家は 1 人もいない**。1 位の「Seen/Unseen 二モデル」をそのまま当コンペに適用する必要はなく、既知批評家用の 1 本のモデルで test 全体をカバーできる。

### 6.2 定義

- **Seen**: test のその行の `critic_name` が、train に 1 回以上登場している。  
- **Unseen**: test のその行の `critic_name` が、train に一度も登場していない。

### 6.3 実装の流れ（unseen が存在するコンペ向け）

当コンペでは unseen＝0 のため不要だが、**別コンペで test に未知批評家が含まれる場合**の参考として記載する。

| モデル | 学習データ | 特徴量 | 予測を出す対象 |
|--------|------------|--------|----------------|
| **既知批評家用** | train 全体 | 現行 55 特徴（`critic_name_te_ts` 等を含む） | test のうち「その行の批評家が seen」の行のみ |
| **未知批評家用** | train 全体 | 批評家 ID 依存を除く or global 置き換え（例: `critic_name` 削除、`critic_name_te_ts` を global 平均などで置換） | test のうち「その行の批評家が unseen」の行のみ |

- 学習: 既知用・未知用の 2 本の LGB をそれぞれ 1 回ずつ学習する。  
- 予測: test の各行について、その行の批評家が seen なら既知用モデルの予測、unseen なら未知用モデルの予測を採用する。  
- 提出: 上記で得た予測を 1 本の CSV（ID 順）にまとめて提出する。

unseen が存在するコンペでは、atmaCup #15 の「CV = seen×0.77 + unseen×0.23」のような**行数重み付き CV** を test の seen/unseen 割合に合わせて再現できる。

---

## 7. 参照

- 改善候補全体: `docs/07_IMPROVEMENT_CANDIDATES.md`  
- 実装: `lib/improvement_candidates.py`（`run_atmacup_ratio`, `run_atmacup_implicit`, `get_bpr_base`）  
- 2-hop 実験: `lib/two_hop.py`（`add_2hop_features`, `run_experiment`）、`train_2hop_experiments.ipynb`  
- 実行: `train_atmacup_5patterns.ipynb`（5 パターン一括）、`train_2hop_experiments.ipynb`（BPR ベース + 2-hop を 1 本ずつ）  
- 参照元: atmaCup #15 1st place solution（ディスカッション）
