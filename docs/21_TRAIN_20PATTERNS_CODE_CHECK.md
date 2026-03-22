# train_20patterns_from_doc20.ipynb コード確認メモ

## EOFError: Ran out of input（embedding 読み込み時）

**原因**: `outputs/embeddings/` 内の embedding 用 `.pkl`（または `.parquet`）が壊れているか、保存が途中で切れている。

**対処**:

1. **壊れたファイルを削除する**
   - `outputs/embeddings/movie_title_info_embeddings.pkl`
   - 同ディレクトリに `movie_title_info_embeddings.parquet` があればそれも削除してよい（再生成で上書きする場合）。

2. **embedding を再生成する**
   - `quick_embedding_submissions.ipynb` で「タイトル+映画情報」の embedding を計算するセルを実行する、または
   - `archive/run_openai_embeddings_title_info_once.py` を実行する。
   - 環境変数 `OPENAI_API_KEY`（または `config/openai_api_key.txt`）が必要。

3. **一時的に embedding なしで動かす**
   - `get_setup(..., use_best_pipeline=False)` にすると、55 特徴ではなくベースライン特徴のみになる。提出スコアは変わるが、ノートブックの動作確認には使える。

---

## 修正した点

- **XGBoost（E）のカテゴリ変換**: 元は `X_tr_e[c].astype("int")` / `X_te_e[c].astype("int")` のみで、train と test で別々に `astype("category")` されていたため、同じ値でもコードが一致しない可能性があった。**train のカテゴリ順で test もエンコード**するように変更（`pd.Categorical(X_te_c[c].astype(object), categories=cats).codes`）。test のみに現れるカテゴリは -1 になる。

## 確認済みで問題なし

1. **セットアップ**
   - `get_setup(use_best_pipeline=True)` → 55 特徴の土台。
   - `get_bpr_base(ctx, factors=64)` → 同じ train/test に BPR64 列を追加したコピーと特徴リストを返す。
   - `add_2hop_features` で `TWO_HOP_REVIEW_COUNT` を追加。4 本目が無い場合は 0.65:0.35 で作成。

2. **A: TF-IDF SVD**
   - `fit_transform(text_tr)` / `transform(text_te)` でリークなし。SVD も train で fit、test は transform のみ。OK。

3. **B: NMF**
   - 同じ `tfidf` で train/test を transform。`MaxAbsScaler` は train で fit。NMF は train で fit、test は transform。非負化してから NMF。OK。

4. **C, D, F: LGB / CatBoost / 浅い LGB**
   - いずれも `X_tr_c` / `X_te_c`（feats_c1 のみ）を使用。LGB は categorical をそのまま扱い、CatBoost は `cat_features` で指定。OK。

5. **ブレンド重み**
   - `paths_5 = [count1, bpr128, stacking, 4th, new]` の順。
   - `rest = 0.55 - w`、`ws = [rest*4/11, rest*4/11, 0.45, rest*3/11, w]` → 合計 1.0。スタッキング 0.45 固定。OK。

6. **blend_n_submissions**
   - `weights` は正規化されるので合計 1.0 でなくても可。`test_ids` で行順を揃え、欠損は 0.5。OK。

7. **提出番号**
   - ある手法の CSV が無い場合は `idx += 2` でスキップするため、01–12 に抜けができる。意図どおり（表の 01–02＝TF-IDF 等は「全手法が生成された場合」の対応）。

## 補足

- **use_label_encoder / eval_metric**: XGBoost 2.0 以降では `use_label_encoder=False` はデフォルト。`eval_metric="logloss"` で二値分類。問題なし。
- **save_submission**: `ctx.test` と予測の長さは一致（いずれも `len(ctx.test)`）。OK。
