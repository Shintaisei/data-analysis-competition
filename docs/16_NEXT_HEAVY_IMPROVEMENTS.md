# 推薦系タスクでまだ試せること（未着手・実装が重め）

現ベース（0.76591）の先に、**推薦タスクとしてまだ手を付けていない／実装が重い**改善をまとめたドキュメント。  
「普通に実装が重そうなやつでいい」という前提で、**未着手 or 実装はあるが現ベースに載せていない**ものを中心に挙げる。

---

## 1. 一覧（実装の重さ・期待度）

| 改善 | 内容 | 現状 | 重さ | 期待 |
|------|------|------|------|------|
| **NN を 1 本アンサンブルに** | 55+BPR64 を MLP で学習し、LGB 予測とブレンド。1位解法で採用。 | 未実装 | 高 | 高（1位で効いた） |
| **BPR64 ベースのスタッキング** | LGB + XGB + CatBoost → Ridge。07 で 55 特徴ベースは実装済み。 | BPR64 の ctx で run_03_stacking は試したが XGB 等でエラー。要修正・再実行。 | 中 | 中 |
| **類似映画を何本レビューしたか（現ベースに載せる）** | 映画 embedding のコサイン類似で top-k 類似映画を求め、その批評家がレビューした本数を 1 列。1位で「特徴量的に一番効いていた」。 | **実装済み** `run_similar_movies_reviewed`。55 特徴ベース or 別提出で 1 本は出せる。**0.76591 の土台（bpr64_count1+BPR128 等）にこの 1 列を足しての提出・ブレンドは未**。 | 低〜中 | 高 |
| **メタの 2-hop（target を使わない）** | 批評家ごとに映画メタ（runtime, genre 等）を集約 → 映画ごとにその集約をさらに集約。1位の「user→item メタ集約」に相当。リークなし。 | 未実装 | 中 | 中〜高 |
| **類似度 TE（似た批評家／似た映画の Fresh 率）** | Jaccard 等で似た批評家・似た映画を求め、その平均 Fresh 率を 1〜2 列。行自身は平均から除く。 | **実装済み** `run_02_similarity_te`。55 特徴ベースでは未提出。**BPR64 / 現ベースに載せての 1 本は未**。 | 中 | 中 |
| **LightGCN** | 接続情報のみで学習する推薦用 GNN。unseen ユーザーには使えない（当コンペは test 批評家 100% train 登場なので可）。 | 未実装。PyTorch Geometric 等が必要。 | 高 | 不明 |
| **独自 GNN（Heterogeneous）** | 批評家・映画の二部グラフで SAGEConv + LayerNorm 等。1位は Heterogeneous GNN で伸ばした。 | 未実装 | 高 | 高（1位で使用） |
| **時系列で行列を切る** | レビュー日で「過去のみ」の接続で (critic, movie) 行列を構築。BPR/ALS の入力としてリークを減らす。 | 未実装。行列構築・検証の手間大。 | 高 | 小〜中 |
| **Surprise（Explicit を 1 特徴に）** | 当コンペは Fresh/Rotten 二値なので、Surprise の SVD 等で「レーティング予測値」を 1 列として LGB に渡す。 | 未実装 | 中 | 不明 |
| **グラフ埋め込み（Node2Vec / ProNE）** | 批評家–映画二部グラフで Node2Vec や ProNE でベクトルを学習し、特徴に追加。 | 07 で候補として記載のみ。実装スキップ。 | 高 | 不明 |
| **2-hop の leave-one-out** | movie_fresh_rate_mean を「その行を除いた」平均で計算し、target リークを除去。2-hop 複数列を再検討できる。 | 未実装 | 中 | 小〜中 |

---

## 2. おすすめの攻め方（優先度イメージ）

### まず試しやすい（実装済み or 土台が近い）

1. **類似映画を何本レビューしたか**  
   `run_similar_movies_reviewed(ctx, top_k=20)` で 1 本出し、**現ベース（0.76591 のブレンド）とさらにブレンド**する。  
   1位で効いていた特徴なので、同じ土台に載せて効くか確認する価値が高い。

2. **類似度 TE**  
   `get_bpr_base(ctx, 64)` で ctx_bpr を作り、`run_02_similarity_te(ctx_bpr)` で BPR64 + 似た批評家/映画の Fresh 率 1〜2 列を 1 本出す。  
   実装はあるので「載せ替え」で済む。

3. **BPR64 ベースのスタッキング**  
   run_03_stacking が BPR64 の ctx で落ちている原因（XGB の eval_set 等）を修正し、55+BPR64 で LGB/XGB/CatBoost→Ridge を 1 本出す。

### 実装が重いが 1 位で効いたもの

4. **NN を 1 本アンサンブルに**  
   55+BPR64 を MLP（または小さい NN）で学習し、LGB 予測とブレンド。1位はこれで伸ばしている。  
   フレームワーク（PyTorch/TF）と CV 設計が必要でコストは高め。

5. **メタの 2-hop**  
   target を使わない 2-hop なので、当コンペで落ちた「Fresh 率・critic_te 平均」の代わりに、映画メタの集約だけを 1〜2 列追加。  
   設計・集約の実装で中コスト。

6. **独自 GNN（Heterogeneous）**  
   批評家・映画の二部グラフで SAGEConv 等。1位解法で使われている。  
   実装・チューニングともに高コスト。

### 独自 GNN を「どこかでモデルをインストール」するか？

**結論: モデルを 1 本インストールするのではなく、GNN 用のライブラリ（レイヤー）を入れて、自分でグラフとモデルを組むイメージです。**

| やること | 内容 |
|----------|------|
| **入れるもの** | **PyTorch** と **PyTorch Geometric（torch_geometric）**。PyG に SAGEConv や Heterogeneous グラフ用の API が入っている。`pip install torch torch_geometric` 等。 |
| **自分で書くもの** | ① train+test の (critic_id, movie_id) から**二部グラフ**を構築（エッジリスト → PyG の `Data` または `HeteroData`）。② 批評家・映画それぞれに**ノードの初期特徴**（ID の embedding や 0）を用意。③ **2〜3 層の SAGEConv + LayerNorm** でノード表現を更新。④ 各 (critic, movie) ペアについて**スコア**（埋め込みの内積 or 1 層 MLP）を出し、train では BCE で学習。⑤ test の (critic, movie) でスコアを計算して提出用の確率に。 |
| **LightGCN の場合** | LightGCN は**論文の 1 モデル**なので、「LightGCN の PyTorch 実装」を GitHub 等から clone／パッケージで入れて、データを渡す形にできる。その場合は「モデルを入れてくる」に近い（RecBole や PyTorch 用の LightGCN 実装など）。 |

つまり **独自 GNN（Heterogeneous）＝ ライブラリで部品を入れて、グラフ構築・学習ループ・予測は自前で書く**。LightGCN だけ試すなら、既存の LightGCN 実装を入れてデータフォーマットを合わせる方が手早い。

### その他（時間があれば）

- **時系列で行列を切る**: BPR/ALS の入力を行列構築時点で「過去のみ」にするとリークが減る。効果はコンペ次第。
- **Surprise**: 二値タスクなので「レーティング予測を 1 特徴に」は要検証。
- **LightGCN / Node2Vec / ProNE**: 推薦ではよく使われるが、当コンペでは未実施。差別化にはなるがコストは高め。

---

## 3. 参照

- 1位解法との対応・全項目: `10_FIRST_PLACE_AND_IMPROVEMENT_REMAINING.md`
- 伸びそうな改善・劇的に: `09_IMPROVEMENT_NEXT.md`
- 実装ありの run_xx 一覧・BPR に載せる: `11_WHAT_NEXT_AFTER_HIGH_POTENTIAL.md`
- 協調フィルタの限界: `15_CF_LIMITS_AND_REMAINING.md`
- 改善候補の詳細（擬似ラベル・TE・スタッキング）: `07_IMPROVEMENT_CANDIDATES.md`
