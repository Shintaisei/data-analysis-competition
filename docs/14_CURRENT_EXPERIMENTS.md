# 現状やっている実験

いま回している実験の**概要・入出力・進め方**をまとめたドキュメント。

---

## 1. どの実験をやっているか

| 項目 | 内容 |
|------|------|
| **ノートブック** | `train_7patterns_from_baseline.ipynb` |
| **ベースライン** | submission_blend_bpr64_bpr128.csv → Public **0.76574** |
| **目的** | ベースライン（BPR64 + BPR128 均等ブレンド）を土台に、重み変更・3 本目追加・シード平均・2-hop 載せの 4 方向で **7 パターン** の提出ファイルを生成する。 |

---

## 2. 実験の流れ（ノートの構成）

1. **セットアップ**  
   `get_setup(seed=42, use_best_pipeline=True)` で 55 特徴のコンテキストを用意。提出先は `outputs/submissions`。

2. **ブレンド用ヘルパー**  
   - `blend_two(path_a, path_b, out_name, w_a, w_b)` … 2 本の加重平均  
   - `blend_three` … 3 本の均等  
   - `blend_three_weighted` … 3 本の任意重み  

3. **単体モデル作成（ここだけ重い）**  
   以下の 4 種類を、**既に CSV がなければ**だけ学習・保存する。  
   - **BPR64 単体** → `submission_2hop_bpr64_only.csv`  
   - **BPR128 単体** → `submission_2hop_bpr128_only.csv`  
   - **ALS64 単体** → `submission_atmacup_implicit_als64.csv`  
   - **BPR64 + 2-hop count1** → `submission_2hop_bpr64_count1.csv`  

4. **パターン 1〜4**  
   上記単体をブレンドして 4 本の提出 CSV を生成。**出力ファイルが既にあればスキップ**。

5. **パターン 5**  
   BPR64 を seed 43, 44 でも学習（seed 42 は bpr64_only で既出）。3 本を均等ブレンドして `submission_blend_bpr64_seed_avg.csv`。**既にあればスキップ**。

6. **パターン 6**  
   bpr64_count1 と BPR128 の均等ブレンド。**既にあればスキップ**。

7. **パターン 7**  
   BPR64 / BPR128 / ALS64 の重み付き (0.35 / 0.45 / 0.2) ブレンド。**既にあればスキップ**。

8. **提出ファイル一覧**  
   7 本の CSV の有無を表示。

---

## 3. 入出力一覧

### 入力（単体モデルの提出 CSV）

| ファイル | 用途 |
|----------|------|
| submission_2hop_bpr64_only.csv | パターン 1, 2, 3, 4, 5, 7 |
| submission_2hop_bpr128_only.csv | パターン 1, 2, 3, 4, 6, 7 |
| submission_atmacup_implicit_als64.csv | パターン 4, 7 |
| submission_2hop_bpr64_count1.csv | パターン 6 |
| submission_2hop_bpr64_only_seed43.csv | パターン 5（seed 43 で別生成） |
| submission_2hop_bpr64_only_seed44.csv | パターン 5（seed 44 で別生成） |

### 出力（7 パターンの提出ファイル）

| # | 提出ファイル | 内容 |
|---|-------------|------|
| 1 | submission_blend_bpr64_bpr128.csv | BPR64 + BPR128 均等（ベースライン再現） |
| 2 | submission_blend_bpr64_bpr128_046.csv | 0.4×BPR64 + 0.6×BPR128 |
| 3 | submission_blend_bpr64_bpr128_064.csv | 0.6×BPR64 + 0.4×BPR128 |
| 4 | submission_blend_bpr64_bpr128_als64.csv | (BPR64 + BPR128 + ALS64) / 3 |
| 5 | submission_blend_bpr64_seed_avg.csv | BPR64 seed 42, 43, 44 の均等平均 |
| 6 | submission_blend_bpr64_count1_bpr128.csv | bpr64_count1 + BPR128 均等 |
| 7 | submission_blend_3way_weighted.csv | 0.35×BPR64 + 0.45×BPR128 + 0.2×ALS64 |

---

## 4. スキップの仕様（再実行・メモリ対策）

- **単体モデル**: 上記 4 種類の CSV が既に `outputs/submissions` に存在する場合は、そのモデルの学習は行わず「既存」と表示。
- **ブレンド 1〜4, 6, 7**: 各パターンの**出力 CSV**が既に存在する場合は、ブレンド計算をせず「既存のためスキップ」と表示。
- **パターン 5**:  
  - seed 43 / 44 用の CSV が既にあればそれらは作らない。  
  - `submission_blend_bpr64_seed_avg.csv` が既にあれば、3 本ブレンドはスキップ。

途中で落ちた場合は、ノートを**先頭から再実行**すれば、できている分はスキップされ、未完了のパターンだけが実行される。

---

## 5. 参照

| ドキュメント | 内容 |
|-------------|------|
| `12_PUBLIC_SCORES_AND_NEW_BASELINE.md` | 過去スコア一覧・新ベースラインの位置づけ |
| `13_7PATTERNS_RATIONALE.md` | 7 パターンの方針・狙い・「上がりそうな理由」 |

現状やっている実験は、この 3 本（12, 13, 14）とノート `train_7patterns_from_baseline.ipynb` を見れば把握できる。
