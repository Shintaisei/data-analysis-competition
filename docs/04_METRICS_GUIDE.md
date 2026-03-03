# 実験指標ガイド — なぜこの指標があるか・結果の見方

`train_metric_driven_experiments.ipynb` の数字の意味と、「なぜこういう評価をしているか」をまとめる。

---

## 1. なぜこの評価指標が生まれたか（経緯）

### 1.1 困っていたこと

- **CV の AUC だけ**を見て特徴量を足していったら、**パブリックでは全部スコアが下がった**（段階的 9 パターン追加は全ボツ）。
- 「CV が良ければ本番も良い」が成り立たず、**何が悪いのか CV の数字だけでは分からなかった**。

### 1.2 やったこと

- 学習済みモデルで **val の予測**を出し、行ごとに **正解（target）・予測（pred）・当たり外れ（TP/TN/FP/FN）** を付けた（`outputs/analysis/train_with_predictions.csv`）。
- それを **セグメント別**（年・ジャンル・top_critic など）に集計した（`outputs/analysis/prediction_summary_by_*.csv`）。

### 1.3 分かったこと

- **弱点セグメント**がはっきりした：  
  `review_year=2015`、`genre_Documentary=1`、`top_critic=True` で AUC が他より低い。
- **予測の偏り**も分かった：  
  全体として「Fresh を出しすぎ」（Rotten を Fresh と誤判定する FP が多い）。  
  つまり **AUC という順位付けはそこそこでも、閾値 0.5 で切ると Fresh に寄りすぎている**。

### 1.4 だからやっていること

- **CV の AUC だけ**で良し悪しを決めない。
- **弱点セグメントの AUC**（`auc_doc_1`, `auc_review_2015`, `auc_topcritic_true`）を一緒に見て、「どこが弱いままか」を追う。
- **FP/FN や pred_pos_rate と pos_rate の差**も見て、「Fresh 出しすぎがひどくなっていないか」を追う。

こうして「**全体 AUC ＋ 弱点セグメント ＋ 偏り**」の 3 つをセットにした評価指標が生まれた。

---

## 2. 結果の見方（具体例で説明）

ノートの実行後、例えば次のような出力が出る：

```
base: auc=0.7602 +- 0.0064, doc_auc=0.7149
publisher_freq: auc=0.7601 +- 0.0070, doc_auc=0.7148
doc_x_critic_te: auc=0.7612 +- 0.0056, doc_auc=0.7179
year_norm_x_critic_te: auc=0.7608 +- 0.0058, doc_auc=0.7150
```

### 2.1 各数字の意味（1 行で何を見ているか）

| 出てくるもの | 意味 |
|--------------|------|
| **auc** | 時系列 CV の 4 fold の **AUC の平均**。従来どおりの「順位付けの良さ」の目安。 |
| **+- の数字** | 4 fold の AUC の**標準偏差**。大きいほど「年によって当たり外れの差が大きい」。 |
| **doc_auc** | **Documentary のセグメント（genre_Documentary=1）だけ**の AUC。ここが元々一番弱かったので、このセグメントが改善しているかを重点的に見る。 |

この 3 つが分かれば、まず「全体の強さ」と「Documentary の強さ」を比べられる。

### 2.2 上記 4 つの結果をどう読むか

- **base**  
  - 全体 AUC = 0.7602、Documentary の AUC = 0.7149。  
  - 比較の**基準**。これより「全体が落ちていないか」「Documentary が極端に悪化していないか」を見る。

- **publisher_freq**  
  - 全体 AUC = 0.7601（base とほぼ同じ）、Documentary = 0.7148（base とほぼ同じ）。  
  - → 効いていない。**採用する理由が薄い**。

- **doc_x_critic_te**  
  - 全体 AUC = **0.7612**（base より **+0.0010**）、Documentary = **0.7179**（base の 0.7149 より **+0.003**）。  
  - → **全体も Documentary も両方良くなっている**。弱点を狙った追加が効いている。**採用候補**。

- **year_norm_x_critic_te**  
  - 全体 AUC = 0.7608（base より少し良い）、Documentary = 0.7150（base とほぼ同じ）。  
  - → 全体は少し伸びているが、Documentary はほとんど変わっていない。**2015 年向け**の改善の可能性はあるが、Documentary 改善という意味では doc_x_critic_te ほどは効いていない。

### 2.3 判断の優先順位（簡易ルール）

1. **まず auc（全体）**  
   base より明らかに落ちていたら、その config は基本的に見送り。

2. **次に doc_auc（Documentary）**  
   全体が少し良くても、doc_auc が base より**悪化**していたら要注意（弱点をさらに弱くしている可能性）。

3. **そのうえで「弱点をちゃんと良くしているか」**  
   全体 AUC が base 以上 **かつ** doc_auc（や 2015 / top_critic のセグメント AUC）が base 以上なら、**採用してパブリックで 1 回試す**価値がある。

今回の例なら、**doc_x_critic_te** が「全体も Documentary も両方改善」なので、まずこれを提出してパブリックスコアを確認する、という流れになる。

---

## 3. 指標一覧（参照用）

実験結果 CSV に出る主な列の意味。

| 指標 | 意味 | 良い方向 |
|------|------|----------|
| auc_mean | 時系列 CV の fold 平均 AUC | 高い |
| auc_std | fold 間の AUC のばらつき | 低い |
| logloss_mean | 確率予測の log loss | 低い |
| brier_mean | 確率と実ラベルの二乗誤差（校正の粗さ） | 低い |
| fp_rate_mean | Rotten を Fresh と誤判定する率 | 低い |
| fn_rate_mean | Fresh を Rotten と誤判定する率 | 低い |
| pos_gap_mean | 予測の Fresh 率 − 実際の Fresh 率（正なら Fresh 出しすぎ） | 0 に近い |
| auc_review_2015 | 2015 年セグメントの AUC | 高い |
| auc_doc_1 | Documentary セグメントの AUC | 高い |
| auc_topcritic_true | top_critic=True セグメントの AUC | 高い |

---

## 4. 実験の流れと出力ファイル

1. `train_metric_driven_experiments.ipynb` を上から実行する。
2. 各 config で時系列 CV が回り、`metric_driven_experiment_results.csv` に上記指標が保存される。
3. **結果の見方**は §2 のとおり。  
   - 全体 AUC が base 以上か  
   - doc_auc（と必要なら 2015 / top_critic）が悪化していないか  
  を確認して、良さそうな config を `SELECTED_CONFIG` に指定し、提出用 CSV を出す。
4. 提出後は**パブリックスコア**で最終判断。CV で良くても Public で落ちる場合は過学習の可能性があるので、変更は小さく抑える。

---

## 5. 実際の結果の解釈例（6 config の場合）

| 順位 | config | auc_mean | doc_auc | 解釈 |
|------|--------|----------|---------|------|
| 1 | publisher_freq__doc_x_critic_te | 0.7616 | 0.7177 | CV で最良。2列追加。 |
| 2 | doc_x_critic_te | 0.7612 | 0.7179 | 1列追加で Documentary も伸びている。 |
| 3 | year_norm_x_critic_te | 0.7608 | 0.7150 | 全体だけ少し伸び、doc は微増。 |
| 4 | base | 0.7602 | 0.7149 | 基準。 |
| 4 | topcritic_x_critic_te | 0.7602 | 0.7149 | base と完全に同じ→列が効いていない。 |
| 6 | publisher_freq | 0.7601 | 0.7148 | ほぼ変化なし。 |

- **効いているのは「doc_x_critic_te」系**（doc 単体と、publisher_freq と組み合わせ）。year_norm は全体のみ微増。topcritic は無効。publisher_freq 単体も無効。
- **CV 上は** `publisher_freq__doc_x_critic_te` が最高だが、列数が多いほど **Public で過学習しやすい**傾向は前回（段階的 9 パターン）でも出ている。

---

## 6. CV は良かったのに Public がちょい下がったとき

### 6.1 起きること

- CV で **auc_mean が base より良く**（例: 0.7616）、**doc_auc も改善**していても、提出すると **Public が base より少し低い**（例: マックス 0.74391、直近 0.74341 で base よりちょい下がり）ことがある。

### 6.2 考えられる理由

1. **過学習**  
   train/val の年度分割では「効く」ように見えても、test（未来の年度・別分布）では効かず、ノイズとして悪影響することがある。**列を足すほど起こりやすい**。
2. **Public の分布**  
   Public の test が、val の年（2013〜2016）と違う年代・傾向だと、CV で良かった特徴が本番では効かない、あるいは逆に効くことがある。
3. **スコアのブレ**  
   「ちょい下がり」程度なら、同じモデルで submit タイミングやランダムシードで揺れる範囲の可能性もある（ただし seed 固定なら小さい）。

### 6.3 試すといいこと

- **ベース（38 列）の提出**を改めて 1 本出し、その Public を「基準」として記録する。
- **doc_x_critic_te だけ**（39 列、1 列追加）の提出も 1 本出して Public を比較する。  
  - 2 列追加（publisher_freq__doc_x_critic_te）より過学習しにくい可能性がある。
- どちらも base より低ければ、**運用は base のまま**にし、特徴量追加は一旦やめて、ハイパラや重みづけなど別の切り口を検討する。

### 6.4 記録しておくとよいこと

- 「publisher_freq__doc_x_critic_te で提出 → Public 0.74341（base よりちょい下がり）」のように、**config 名と Public スコア**を 03 や 04 に 1 行メモしておくと、次に同じことを繰り返さずに済む。

---

## 7. まとめ

- **なぜこの指標があるか**：CV だけだと Public で全滅したので、「どこが弱いか」「どれだけ Fresh に偏っているか」を数値で追うために、セグメント別 AUC と FP/FN・pos_gap を入れた。
- **結果の見方**：  
  - まず **auc** で全体が base 以上か。  
  - 次に **doc_auc**（と 2015 / top_critic）で弱点が悪化していないか・改善しているか。  
  - そのうえで「全体も弱点も良くなっている」config を採用候補にしてパブリックで試す。
- **CV 良くても Public がちょい下がることはある**（過学習・分布差）。そのときは **base 提出**と **1 列だけ追加（doc_x_critic_te）** を出して比較し、どちらも base より低ければ運用は base のままにする。
