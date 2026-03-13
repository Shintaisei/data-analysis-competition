# ベースラインコード検証

本番ベースライン **Public 0.75493**（doc_x_critic_te）および `train_baseline.ipynb` の根幹が正しいことを確認した記録。

**ベースライン更新**: 協調フィルタ（implicit BPR）で更新。BPR 16 で 0.76101、**BPR 64（submission_2hop_bpr64_only.csv）で Public 0.76479** を達成。**現在のベースライン（超える目標）は 0.76479**。詳細は `docs/08_COLLABORATIVE_FILTERING.md` §2.5・§5 の表を参照。次に試すべきこと（伸びそうな改善）は **`docs/09_IMPROVEMENT_NEXT.md`** にまとめてある。

---

## 1. 本番ベースライン 0.75493 を出すパイプライン

**実装場所**: `quick_embedding_submissions.ipynb` の「# 7. doc_x_critic_te」ブロック

| 項目 | 期待値 | 実装 |
|------|--------|------|
| データ | `get_baseline_data()` の train / test | ✅ 同じ `train, test` を全提出で共有 |
| Embedding | movie_title_info（small） | ✅ `EMBEDDING_NAME = "movie_title_info"`, `load_embeddings` → `_merge_embeddings` |
| 次元削減 | PCA 16, random_state=42 | ✅ `fit_transform_embedding(..., method="pca", n_components=16, random_state=42)` |
| ベース特徴量 | lib の 38 特徴（BASELINE_FEATURES） | ✅ `base_features = list(BASELINE_FEATURES)` |
| 追加 1 列 | critic_te × genre_Documentary | ✅ `critic_name_te_ts` と `genre_Documentary` は両方 `get_baseline_data()` で用意される（pipeline + create_features） |
| モデル | LGB, 全 train で 1 本学習 | ✅ `LGBMClassifier(**BASELINE_LGB_PARAMS)`, `model.fit(X_tr[feats], y)` |
| 提出 | test の行順で予測・保存 | ✅ `save_submission(test, pred, path, sanitize=True)`（pred は X_te の行順＝test の行順） |

- `genre_Documentary`: `feature_engineering.create_features` で `genres` から生成（lib.pipeline は `create_features` を経由）。
- `critic_name_te_ts`: `lib.pipeline.add_3c3_and_text_meta` で時系列 TE として追加。
- 両方とも `lib/pipeline.py` の `BASELINE_FEATURES` に含まれる（38 特徴のうちの 1 つずつ）。

---

## 2. train_baseline.ipynb（メタのみ 38 特徴）

| 項目 | 期待値 | 実装 |
|------|--------|------|
| データ | get_baseline_data() | ✅ `train, test = get_baseline_data()` |
| 特徴量 | 38（BASELINE_FEATURES） | ✅ `FEATURES = BASELINE_FEATURES`, 出力 "Features: 38" と一致 |
| 検証 | 時系列 CV（2013〜2016 を val） | ✅ VAL_YEARS = [2013,2014,2015,2016], review_year で分割 |
| 提出用 | 全 train で 1 本学習 → test 予測 | ✅ `model_full.fit(X, y)`, `final_pred = model_full.predict_proba(X_test)[:, 1]` |
| LGB パラメータ | BASELINE_LGB_PARAMS | ✅ `lgb_params = BASELINE_LGB_PARAMS` |

- 本ノートは「メタのみ」ベースライン。本番で超える目標スコアは **0.75493** と説明セルに明記済み。

---

## 3. lib の一貫性

| 項目 | 内容 |
|------|------|
| BASELINE_FEATURES | `lib/pipeline.py` で 38 個定義。`genre_Documentary`, `critic_name_te_ts` を含む。 |
| get_baseline_data | `load_train_test` → `create_features` → `prepare_baseline_data`。両ノートともここだけ使えば同じ train/test。 |
| BASELINE_LGB_PARAMS | `lib/pipeline.py` で定義。quick_embedding / train_baseline とも同じものを使用。 |
| save_submission | `lib/submission.py`。pred 長さ・ID 一致をチェックし、sanitize で [0,1] にクリップ。 |

---

## 4. 結論

- **本番ベースライン 0.75493** を出すコード（doc_x_critic_te）は、上記の通りデータ・特徴量・PCA・LGB・提出まで整合している。
- **train_baseline.ipynb** は 38 特徴・時系列 CV・提出用 1 本学習まで、説明とコードが一致している。
- 今後の改善実験は、土台は同じ `get_baseline_data` / `BASELINE_FEATURES` / `BASELINE_LGB_PARAMS` のまま、**超える目標は現時点の最高 0.76479**（submission_2hop_bpr64_only.csv、BPR 64、08 協調フィルタ §2.5）とする。0.75493 / 0.76101 は従来のベースラインとして参照用に残す。
