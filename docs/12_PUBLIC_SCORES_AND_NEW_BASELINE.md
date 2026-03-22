# Public Score 提出結果と新ベースライン

協調フィルタ・2-hop・ブレンド系の提出結果を一覧化し、**最高スコアを新ベースライン**としてここから仕切り直すためのドキュメント。

---

## 1. 最高スコア（更新済み）

| 提出ファイル | Public Score | 備考 |
|-------------|--------------|------|
| **submission_baseline_improve_03.csv** | **0.76971** | **現時点の最高**。ベースライン改善6パターン（5本目＝新モデル TF-IDF SVD 10%）。 |
| submission_baseline_improve_04.csv | 0.76970 | 同上・新モデル 12%。 |
| submission_baseline_improve_02.csv | 0.76970 | 同上・新モデル 8%。 |
| submission_baseline_improve_05.csv | 0.76970 | 同上・新モデル 15%。 |
| submission_baseline_improve_01.csv | 0.76969 | 同上・新モデル 5%。 |
| submission_baseline_improve_06.csv | 0.76968 | 同上・新モデル 18%。 |
| submission_blend_4way_020_020_045_015.csv | 0.76967 | 旧最高。4本ブレンド・4本目 15%（count1:BPR128:stacking:4本目=0.20:0.20:0.45:0.15）。 |
| submission_blend_4way_020_020_050_010.csv | 0.76956 | 4本ブレンド・4本目 10%。 |
| submission_blend_3way_030_020_050.csv | 0.76958 | 3本: count1 重め・BPR128 軽め・stacking 50%。 |
| submission_blend_3way_025_025_050.csv | 0.76955 | 3本: 0.25:0.25:0.5（旧最高）。 |
| submission_blend_3way_023_027_050.csv | 0.76953 | 3本: 微調整。 |
| submission_blend_3way_020_030_050.csv | 0.76949 | 3本: count1 軽め・BPR128 重め。 |
| submission_blend_3way_equal.csv | 0.76945 | 3本均等 (1/3, 1/3, 1/3)。 |
| submission_blend_3way_022_023_055.csv | 0.76934 | 3本: stacking 55%。 |
| submission_blend_3way_040_030_030.csv | 0.76934 | count1 やや多め。 |
| submission_blend_3way_050_025_025.csv | 0.76904 | count1 多め。 |
| submission_blend_3way_025_050_025.csv | 0.76860 | BPR128 多め。 |
| submission_blend_stacking_bpr128_w065.csv | 0.76820 | スタッキング : BPR128 = 0.65 : 0.35。 |
| submission_blend_ratio_w065.csv | 0.76609 | 2本最高。count1 : BPR128 = 0.65 : 0.35。 |
| submission_blend_ratio_w055.csv | 0.76599 | count1 : BPR128 = 0.55 : 0.45。 |
| submission_blend_bpr64_count1_bpr128.csv | 0.76591 | count1 + BPR128 0.65:0.35（7パターン #6）。 |
| submission_blend_bpr64_bpr128_064.csv | 0.76588 | 0.6×BPR64 + 0.4×BPR128（7パターン #3）。 |
| submission_blend_bpr64_seed_avg.csv | 0.76578 | BPR64 seed 42,43,44 平均（7パターン #5）。 |
| submission_blend_bpr64_bpr128.csv | 0.76574 | 旧ベースライン。BPR64 + BPR128 均等。 |
| submission_improvement_03_stacking.csv | 0.76177 | スタッキング単体（3本ブレンドの入力の1本）。 |

**ベースライン改善6パターン（5本目＝BPR64+count1+TF-IDF SVD の新モデルを 5%〜18% でブレンド）で 0.76968〜0.76971 を記録。最高は 03（新モデル 10%）の 0.76971。前最高 0.76967 から微小な改善にとどまり、ブレンド・スタッキング系は 0.77 前後で頭打ち。特徴量追加や別手法が次の一手。**

### 1.1 ベースライン改善 6 パターン（train_baseline_improve_6patterns.ipynb）

| 提出ファイル | Public Score | 5本目（新モデル）の重み |
|-------------|--------------|-------------------------|
| submission_baseline_improve_03.csv | **0.76971** | 10% |
| submission_baseline_improve_04.csv | 0.76970 | 12% |
| submission_baseline_improve_02.csv | 0.76970 | 8% |
| submission_baseline_improve_05.csv | 0.76970 | 15% |
| submission_baseline_improve_01.csv | 0.76969 | 5% |
| submission_baseline_improve_06.csv | 0.76968 | 18% |

※ 新モデル＝BPR64+count1+TF-IDF SVD → LGB。既存4本＋この1本でスタッキング 0.45 固定。**微小に上がっただけ**で、ブレンド重みの違いによる差は小さい。

---

## 2. 提出結果一覧（Public Score 降順）

### 2.1 ブレンド系（上位）

| 提出ファイル | Public Score |
|-------------|--------------|
| submission_blend_bpr64_bpr128.csv | **0.76574** |
| submission_blend_bpr64_als64.csv | 0.76451 |

### 2.2 2-hop 単体・BPR 系

| 提出ファイル | Public Score |
|-------------|--------------|
| submission_2hop_bpr64_count1.csv | 0.76511 |
| submission_2hop_bpr64_only.csv | 0.76479 |
| submission_2hop_bpr64_ratio_count1.csv | 0.76177 |
| submission_2hop_bpr128_only.csv | 0.76306 |
| submission_2hop_axis1_bpr256.csv | 0.76368 |
| submission_2hop_axis1_bpr128_iter200.csv | 0.76265 |
| submission_2hop_axis1_bpr128_reg_low.csv | 0.76292 |
| submission_2hop_axis1_bpr128.csv | 0.76306 |
| submission_2hop_bpr32_only.csv | 0.76366 |
| submission_2hop_bpr_only.csv | 0.76101 |
| submission_2hop_bpr_count.csv | 0.76111 |
| submission_2hop_bpr_critic_te.csv | 0.75798 |
| submission_2hop_bpr_2hop_ratio.csv | 0.75752 |
| submission_2hop_axis1_bpr256_reg_high.csv | 0.75718 |
| submission_2hop_axis1_bpr128_reg_high.csv | 0.75609 |
| submission_2hop_bpr_plus_dot.csv | 0.76042 |
| submission_2hop_bpr_dot_only.csv | 0.75331 |
| submission_2hop_bpr32_2hop_ratio.csv | 0.76073 |
| submission_2hop_bpr_fresh.csv | 0.72223 |
| submission_2hop_bpr_all2hop.csv | 0.72594 |
| submission_2hop_bpr_all2hop_ratio.csv | 0.72471 |
| submission_2hop_bpr_all2hop_plus_dot.csv | 0.72364 |

### 2.3 Implicit ALS 単体

| 提出ファイル | Public Score |
|-------------|--------------|
| submission_atmacup_implicit_als64.csv | 0.75867 |

### 2.4 軸・正則化などのバリエーション

（上表の axis1 / reg_high / reg_low / iter200 等に含む。）

---

## 3. 傾向の整理

- **ブレンドが最良**: BPR64 + BPR128 のブレンドが 0.76574 で最高。BPR64 + ALS64 も 0.76451 と高い。
- **2-hop 単体**: bpr64_count1（0.76511）、bpr64_only（0.76479）が 2-hop 系のなかでは上位。
- **all2hop / fresh**: 0.72 台で大きく落ちており、当コンペでは逆効果。
- **正則化**: reg_high はやや悪化、reg_low は bpr128 と同程度。axis1 の bpr256 が 0.76368 でまずまず。

---

## 4. 新ベースラインからの進め方

1. **ベース**: **submission_blend_bpr64_bpr128.csv（0.76574）** を「今の最高」として扱う。
2. **新ノートブック**: この 0.76574 を再現するパイプライン（BPR64 + BPR128 のブレンド）を 1 本目に組み、その上で以下を試す。
   - ブレンド重みの調整（均等でない加重）
   - BPR64 + BPR128 + ALS64 の 3 本ブレンド
   - シード平均（BPR64/BPR128 を複数 seed で学習して平均）
   - 2-hop や count1 / ratio_count1 を「ブレンドベース」に載せる
3. **参照ドキュメント**  
   - 改善候補の一覧・優先度: `09_IMPROVEMENT_NEXT.md`、`10_FIRST_PLACE_AND_IMPROVEMENT_REMAINING.md`  
   - 協調フィルタの詳細: `08_COLLABORATIVE_FILTERING.md`  
   - high potential のあとにやれること: `11_WHAT_NEXT_AFTER_HIGH_POTENTIAL.md`

---

## 5. まとめ

| 項目 | 内容 |
|------|------|
| **現時点の最高** | **submission_blend_4way_020_020_045_015.csv** → **Public 0.76967**（4本ブレンド・4本目 15%） |
| **何が効いたか** | 3本（count1 + BPR128 + stacking）に **4本目（2本ブレンド or similar/improvement_05）を 15% で追加** すると 0.76967 まで伸びた。 |
| **頭打ちの認識** | ブレンド・スタッキング系は 0.769 台で伸び悩み。**特徴量追加**（新規 2-hop・類似度・メタ集約等）や **別手法**（NN・GNN・重い LGB 等）が次の一手。 |
| **2本ブレンド最高** | submission_blend_ratio_w065.csv → 0.76609。 |
| **避けるパターン** | all2hop / fresh 系（0.72 台で悪化） |

---

## 5.5 ブレンド比率探索の結果（train_priority_blend_experiments）

count1 : BPR128 の比率を変えた提出（同一の 2 本をブレンドするだけ）。0.5:0.5 より **count1 を重くする** とスコアが伸びた。

| 提出ファイル | 比率 (count1 : BPR128) | Public Score |
|-------------|------------------------|--------------|
| submission_blend_ratio_w065.csv | 0.65 : 0.35 | **0.76609** |
| submission_blend_ratio_w060.csv | 0.60 : 0.40 | 0.76606 |
| submission_blend_ratio_w055.csv | 0.55 : 0.45 | 0.76599 |
| submission_blend_ratio_w050.csv | 0.50 : 0.50 | 0.76591 |
| submission_blend_ratio_w040.csv | 0.40 : 0.60 | 0.76562 |

→ **次の一手**: このブレンド構成（左側 1 本 + 右側 BPR128）を維持したまま、**左側をより高精度モデル**（スタッキング・NN・重い LGB 等）に差し替え、同じく 0.65:0.35 前後の比率で試す。

---

## 5.6 3本ブレンドの結果（train_priority_blend_experiments・2025年3月）

count1 + BPR128 + stacking の 3 本で重みを変えた提出。**スタッキングを 50% にした 0.25:0.25:0.5 が 0.76955 で最高記録を更新。**

| 提出ファイル | 重み (count1 : BPR128 : stacking) | Public Score |
|-------------|----------------------------------|--------------|
| submission_blend_3way_025_025_050.csv | 0.25 : 0.25 : **0.50** | **0.76955** |
| submission_blend_3way_equal.csv | 1/3 : 1/3 : 1/3 | 0.76945 |
| submission_blend_3way_040_030_030.csv | 0.40 : 0.30 : 0.30 | 0.76934 |
| submission_blend_3way_050_025_025.csv | 0.50 : 0.25 : 0.25 | 0.76904 |
| submission_blend_3way_025_050_025.csv | 0.25 : 0.50 : 0.25 | 0.76860 |

- スタッキング単体: submission_improvement_03_stacking.csv → 0.76177
- スタッキング + BPR128 (0.65:0.35): submission_blend_stacking_bpr128_w065.csv → 0.76820

→ **スタッキングを 3 本のうち 50% にすると、2 本ブレンド最高 0.76609 を約 +0.003 上回った。**

### 5.7 4本ブレンドの結果（train_best_based_7patterns 等）

count1 + BPR128 + stacking + 4本目（2本ブレンド or similar/improvement_05）で重みを変えた提出。**4本目 15% が 0.76967 で最高。**

| 提出ファイル | 重み（4本目） | Public Score |
|-------------|---------------|--------------|
| submission_blend_4way_020_020_045_015.csv | 0.20 : 0.20 : 0.45 : **0.15** | **0.76967** |
| submission_blend_4way_020_020_050_010.csv | 0.20 : 0.20 : 0.50 : 0.10 | 0.76956 |

→ ブレンド・スタッキング系はここが頭打ちの可能性。次の一手は**特徴量追加**（類似映画・メタ2-hop・TF-IDF等）や**別モデル**（NN・GNN・重いLGB）。

---

## 6. 7パターン実験の結果（提出後）

`train_7patterns_from_baseline.ipynb` で実行した 7 パターンの Public Score。提出日時順の結果をスコア降順で整理。

### 6.1 7パターン本番（ブレンド結果）

| 順位 | 提出ファイル | Public Score | パターン |
|------|-------------|--------------|---------|
| 1 | submission_blend_bpr64_count1_bpr128.csv | **0.76591** | #6: bpr64_count1 + BPR128 均等 |
| 2 | submission_blend_bpr64_bpr128_064.csv | 0.76588 | #3: 0.6×BPR64 + 0.4×BPR128 |
| 3 | submission_blend_bpr64_seed_avg.csv | 0.76578 | #5: BPR64 seed 42,43,44 平均 |
| 4 | submission_blend_bpr64_bpr128.csv | 0.76574 | #1: ベースライン再現（均等） |
| 5 | submission_blend_3way_weighted.csv | 0.76550 | #7: 0.35×BPR64 + 0.45×BPR128 + 0.2×ALS64 |
| 6 | submission_blend_bpr64_bpr128_046.csv | 0.76548 | #2: 0.4×BPR64 + 0.6×BPR128 |
| 7 | submission_blend_bpr64_bpr128_als64.csv | 0.76520 | #4: 3本均等（BPR64+BPR128+ALS64） |

### 6.2 単体・シード別（7パターンの入力）

| 提出ファイル | Public Score | 備考 |
|-------------|--------------|------|
| submission_2hop_bpr64_only.csv | 0.76479 | BPR64 単体（seed 42） |
| submission_2hop_bpr64_only_seed44.csv | 0.76547 | BPR64 seed 44（#5 用） |
| submission_2hop_bpr64_only_seed43.csv | 0.76537 | BPR64 seed 43（#5 用） |
| submission_2hop_bpr64_count1.csv | 0.76511 | BPR64 + 2-hop count1（#6 用） |
| submission_2hop_bpr128_only.csv | 0.76306 | BPR128 単体 |
| submission_atmacup_implicit_als64.csv | 0.75867 | ALS64 単体 |

### 6.3 所感

- **#6（count1 + BPR128）が最高**: 2-hop の movie_review_count 1 列を載せた bpr64_count1 と BPR128 の均等ブレンドが 0.76591 でベースライン 0.76574 を上回った。
- **#3（0.6/0.4）が 2 位**: BPR64 をやや重くしたブレンドも 0.76588 と僅差で良い。
- **#5（シード平均）**: BPR64 のみのシード平均で 0.76578。単体 0.76479 より約 +0.001 の改善。
- **#4（3本均等）**: ALS を足すと 0.76520 で、2本ブレンドよりやや落ちる。ALS 単体が 0.75867 と低いため、均等に混ぜると足を引っ張る形になったと考えられる。
