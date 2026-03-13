# 伸びそうな改善一覧

**ベースライン（超える目標）**: Public **0.76479**（`submission_2hop_bpr64_only.csv`：55 特徴 + BPR 64 の 128 列。2-hop なし）。

以下は **スコアが伸びそうな改善**を優先度・実装コストでまとめたもの。新規実験は 1 本ずつ提出してベースラインを下回らないか確認する。詳細な根拠や既存結果は `docs/08_COLLABORATIVE_FILTERING.md` を参照。

---

## 1. 優先度順：試すべき改善

| 優先度 | 改善内容 | 理由・期待 | やり方・実装 | コスト |
|--------|----------|------------|--------------|--------|
| **1** | **ALS factors=64 を 1 本出す** | BPR と目的関数が違うので ALS 64 単体が効く可能性。既存コードで実行するだけ。 | `run_atmacup_implicit(ctx, "als", 64, "als64")`（`train_atmacup_5patterns.ipynb` や新規ノートで実行）。 | 低 |
| **2** | **BPR 64 + movie_review_count の 1 列だけ** | bpr_count は BPR 16 で微増(0.76111)。2-hop は 1 列（リークなしのレビュー数）だけなら悪化を抑えられる可能性。 | `run_experiment(ctx, "bpr64_count1", bpr_factors=64, use_2hop_cols=[TWO_HOP_REVIEW_COUNT])`（`lib.two_hop.TWO_HOP_REVIEW_COUNT`）。 | 低 |
| **3** | **BPR 64 と ALS 64 のブレンド** | 行列分解の違いが補正になり、単体より伸びる可能性。まず各単体のスコアを出してから。 | 提出 CSV 2 本の予測を読み、加重平均で 1 本作成。重みは均等 or CV で探索。`run_quick_embedding_submissions.py` の ensemble を参考に。 | 低 |
| **4** | **BPR 64 と BPR 128 のブレンド** | 単体では BPR 64 の方が高いが、2 本の平均で安定・微増の可能性。 | 同上。 | 低 |
| **5** | **confidence 重み（ALS）** | 接続を 0/1 ではなくレビュー数等で重み付け。implicit の ALS は重み付き行列をサポート。 | `lib/improvement_candidates.py` の `_build_implicit_embeddings` で行列構築を拡張（`data` を重みに）。ALS 用のみ。 | 中 |
| **6** | **NMF / TruncatedSVD（CF 行列）** | BPR と異なる分解で相補的な信号の可能性。1 本ずつ試す。 | 同じ (critic, movie) 0/1 行列を `sklearn.decomposition.NMF` または `TruncatedSVD` で分解し、批評家・映画ベクトルを特徴として追加する関数を新規追加。 | 中 |
| **7** | **ALS のハイパーパラメータ** | regularization, iterations を変えて安定・微増の可能性。 | `_build_implicit_embeddings` に `als_kwargs` を渡す拡張（BPR の `bpr_kwargs` と同様）。 | 中 |
| **8** | **時系列で行列を切る** | レビュー日で「過去のみ」の接続で行列を作り、リークを減らす。 | 行列構築時に日付フィルタを追加。実装・検証コスト高。 | 高 |

---

## 2. すぐ試せるもの（コード変更ほぼ不要）

- **ALS 64**: 上記 1。既存 `run_atmacup_implicit` の第 3 引数を 64 にするだけ。
- **BPR 64 + movie_review_count 1 列**: 上記 2。既存 `run_experiment` で `use_2hop_cols=[TWO_HOP_REVIEW_COUNT]`, `bpr_factors=64` を指定するだけ。
- **ブレンド**: 上記 3・4。既存提出 CSV を 2 本読み、予測を加重平均して 1 本保存するスクリプト or ノート 1 セル。

---

## 3. 試さない方がよさそうなもの（現状の知見）

- **BPR の次元を 128 / 256 に増やす** … 軸1で実施済み。いずれも 0.76479 未満（256 が 0.76368 で最高）。
- **2-hop を 3 列まとめて足す** … Fresh 率平均・critic_te 平均・レビュー数をまとめて足すとスコアが大きく落ちた（0.72 台）。
- **2-hop の Fresh 率平均・割り算を足す** … target リークや過学習で落ちている。leave-one-out 等の修正をしない限り避ける。

---

## 4. 参照

- 協調フィルタの詳細・結果表・軸の説明: `docs/08_COLLABORATIVE_FILTERING.md`（§5.5〜§5.7）
- ベースライン検証: `docs/BASELINE_VERIFICATION.md`
- 実装: `lib/improvement_candidates.py`（`run_atmacup_implicit`, `get_bpr_base`）、`lib/two_hop.py`（`run_experiment`, `TWO_HOP_REVIEW_COUNT`）
