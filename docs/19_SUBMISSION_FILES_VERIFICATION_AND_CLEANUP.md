# 提出ファイルのコード検証と削除の目安

## 1. コード検証結果（すべて正しい）

`lib/submission.py` の仕様:
- **blend_two_submissions(path_a, path_b, out_path, weight_a)**  
  → 結果 = `weight_a * a + (1 - weight_a) * b`（path_a が weight_a）
- **blend_n_submissions(paths, weights, out_path)**  
  → weights は paths の順に対応し、内部で正規化（合計 1.0）される。

### 各ファイルと重みの対応

| ファイル | 意図した重み | コード上の確認 |
|----------|--------------|----------------|
| submission_improvement_03_stacking.csv | スタッキング単体 | run_03_stacking の出力。✓ |
| submission_2hop_bpr64_count1.csv | count1 単体 | run_experiment("bpr64_count1"). ✓ |
| submission_2hop_bpr128_only.csv | BPR128 単体 | run_experiment("bpr128_only"). ✓ |
| submission_blend_ratio_w035.csv 〜 w065.csv | count1 : BPR128 = w : (1-w) | path_count1, path_bpr128, weight_a=0.35〜0.65。✓ |
| submission_blend_bpr64_count1_bpr128.csv | count1 : BPR128 = 0.65 : 0.35 | blend_two(path_count1, path_bpr128, weight_a=0.65)。✓ |
| submission_blend_stacking_bpr128_w065.csv | stacking : BPR128 = 0.65 : 0.35 | blend_two(path_stacking, path_bpr128, weight_a=0.65)。✓ |
| submission_blend_3way_equal.csv | 1/3 : 1/3 : 1/3 | paths=[count1,bpr128,stacking], weights=[1,1,1]。✓ |
| submission_blend_3way_040_030_030.csv | 0.4 : 0.3 : 0.3 | weights=[0.4, 0.3, 0.3]。✓ |
| submission_blend_3way_050_025_025.csv | 0.5 : 0.25 : 0.25 | weights=[0.5, 0.25, 0.25]。✓ |
| submission_blend_3way_025_050_025.csv | 0.25 : 0.5 : 0.25 | weights=[0.25, 0.5, 0.25]。✓ |
| submission_blend_3way_025_025_050.csv | 0.25 : 0.25 : 0.5（現最高） | weights=[0.25, 0.25, 0.5]。✓ |

**結論: コードとファイル名・重みの対応に不整合はありません。**

---

## 2. 消さないほうがいいファイル（必須）

これらが無いとブレンドを**再生成できない**か、**再計算に時間がかかる**ため残す推奨。

| ファイル | 理由 |
|----------|------|
| **submission_2hop_bpr64_count1.csv** | 3本・2本ブレンドの「元データ」の 1 本。 |
| **submission_2hop_bpr128_only.csv** | 同上。 |
| **submission_improvement_03_stacking.csv** | 同上。スタッキングは学習に時間がかかる。 |
| **submission_blend_bpr64_count1_bpr128.csv** | 参照用ベスト（2本）。他ノート・スクリプトが参照する可能性。 |
| **submission_blend_3way_025_025_050.csv** | 現最高提出。記録・再提出用に保持推奨。 |

---

## 3. 消してもよいファイル（必要なら削除可）

上記 3 本（count1, BPR128, stacking）が揃っていれば、ノートブックで**数秒で再生成**できる。

| ファイル | 再生成方法 |
|----------|------------|
| submission_blend_ratio_w035.csv 〜 w065.csv | `train_priority_blend_experiments` の §1 または blend_two を該当 weight_a で実行。 |
| submission_blend_stacking_bpr128_w065.csv | blend_two(path_stacking, path_bpr128, weight_a=0.65)。 |
| submission_blend_3way_equal.csv | blend_n(paths_3, [1,1,1], ...)。 |
| submission_blend_3way_040_030_030.csv | blend_n(paths_3, [0.4,0.3,0.3], ...)。 |
| submission_blend_3way_050_025_025.csv | blend_n(paths_3, [0.5,0.25,0.25], ...)。 |
| submission_blend_3way_025_050_025.csv | blend_n(paths_3, [0.25,0.5,0.25], ...)。 |

**まとめ**: 容量を減らしたい場合は「3.」を削除してよい。**「2.」の 5 つは消さないほうがよい。**
