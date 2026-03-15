# train_high_potential_experiments のあとにまだやれること

このノートで一通り提出ファイルを出した**あと**に試せることをまとめたもの。ノートは触らずここだけ見ればよい。詳細な根拠・全項目は `10_FIRST_PLACE_AND_IMPROVEMENT_REMAINING.md` と `09_IMPROVEMENT_NEXT.md` を参照。

---

## このノートでやっていること（済）

- ベースライン（bpr64_only）・ALS64・bpr64_count1・bpr64_ratio_count1・bpr128
- ブレンド 2 種（BPR64+ALS64、BPR64+BPR128）
- BPR64 スタッキング（LGB+XGB+CatBoost→Ridge）
- 類似映画を何本レビューしたか（run_similar_movies_reviewed）

---

## まだやれること（どこらへんか）

### 実装あり・BPR64 に載せてないだけ

| 内容 | やり方 |
|------|--------|
| 類似度 TE（02） | `run_02_similarity_te(ctx)` を BPR64 の train/test で呼ぶ。ctx を get_bpr_base で差し替えた ctx_bpr で run_02 を実行するか、BPR64 特徴＋te_sim_critic, te_sim_movie で 1 本出す。 |
| 擬似ラベル閾値見直し（01） | run_01_pseudo_label。0.98/0.02 や件数キャップ・sample_weight を変えて BPR64 ベースで 1 本。 |
| scale_pos_weight（05） | run_05_scale_pos_weight。BPR64 の ctx_bpr で実行。 |
| 特徴選択（07） | run_07_feature_selection。BPR64 の ctx_bpr で実行。 |
| 弱点 1 列（10） | run_10_extra_col。BPR64 の ctx_bpr で実行。 |
| TF-IDF+SVD（08） | run_08_tfidf_svd。BPR64 の ctx_bpr で実行。 |

→ いずれも `lib.improvement_candidates` に実装あり。別ノートで ctx = get_setup(use_best_pipeline=True) → get_bpr_base(ctx, 64) で ctx_bpr を作り、各 run_xx(ctx_bpr) を 1 本ずつ出せばよい。

### ブレンド・アンサンブル（コード変更ほぼ不要）

- **重み付きブレンド**: 均等ではなく、CV スコアや逆相関で重みを決めて 2 本以上を加重平均。09 に記載。
- **シード平均**: BPR64 を seed 42, 43, 44... で学習し、予測を平均して 1 本。quick_embedding で実績あり。BPR64 ベースでは未。

### 協調フィルタの拡張（lib 拡張）

- **confidence 重み（ALS）**: 接続を 0/1 ではなくレビュー数等で重み付け。`_build_implicit_embeddings` で行列を拡張。中コスト。
- **NMF / TruncatedSVD**: 同じ (critic, movie) 行列を sklearn で分解し、批評家・映画ベクトルを特徴に。新関数追加。中コスト。
- **ALS ハイパーパラメータ**: als_kwargs を渡す拡張（bpr_kwargs と同様）。中コスト。
- **時系列で行列を切る**: レビュー日で「過去のみ」の接続で行列を作る。高コスト。

### 2-hop の修正版（当コンペで落ちた部分の対策）

- **leave-one-out**: movie_fresh_rate_mean を「その行を除いた」平均で計算し、target リークをなくす。未実装。中コスト。
- **メタのみ 2-hop**: Fresh 率や critic_te ではなく、映画のメタ（runtime, genre 等）を「その映画をレビューした批評家」で集約した値だけを使う。target を使わない。未実装。中コスト。

### 劇的に（未実装）

- **メタの 2-hop**: 上に同じ。1 位で効いていたパターン。
- **NN を 1 本アンサンブルに**: 55+BPR64 を MLP で学習し、LGB とブレンド。高コスト。

### 評価・インフラ

- **CV 加重**: 時系列 AUC と GroupKFold AUC を 0.7/0.3 等で加重してモデル比較。低コスト。
- **特徴量 feather 保存・wandb/hydra**: 前処理キャッシュ、実験管理。中コスト。

---

## 参照

- 全項目・1 位解法との対応: `docs/10_FIRST_PLACE_AND_IMPROVEMENT_REMAINING.md`
- 伸びそうな改善の優先度: `docs/09_IMPROVEMENT_NEXT.md`
- 協調フィルタの詳細: `docs/08_COLLABORATIVE_FILTERING.md`
