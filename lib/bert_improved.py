"""
BERT/ModernBERT/DeBERTa で (critic, movie) 行のテキストを分類し、提出用予測を出す。

- デフォルトは ModernBERT（answerdotai/ModernBERT-base）。atmaCup 上位でよく使われる。
- 1位共有は DeBERTa v3 large（microsoft/deberta-v3-large）で LB 0.76242。
- 参照実装に合わせて FILL_MAP（欠損を [NO_*] に）、StratifiedKFold、WeightedTrainer（pos_weight）、
  DataCollatorWithPadding、warmup/cosine/weight_decay/bf16 をサポート。

テキスト構築（1位共有フォーマット）:
  [MOVIE] title [INFO] movie_info [GENRES] ... [DIRECTORS] ... [PRODUCTION] ... [CRITIC] ... [TOP_CRITIC] ... [PUBLISHER] ... [CONTENT] ...
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .submission import save_submission, verify_submission

# 参照実装: 欠損をプレースホルダーに（トークナイザーで区別しやすくする）
FILL_MAP = {
    "movie_title": "[NO_TITLE]",
    "movie_info": "[NO_INFO]",
    "genres": "[NO_GENRES]",
    "directors": "[NO_DIRECTORS]",
    "production_company": "[NO_PRODUCTION]",
    "critic_name": "[NO_CRITIC]",
    "publisher_name": "[NO_PUBLISHER]",
    "content_rating": "[NO_RATING]",
}


def build_bert_text(df: pd.DataFrame, use_fill_map: bool = True) -> pd.Series:
    """
    1位共有フォーマットで行ごとのテキストを構築。
    use_fill_map=True のとき欠損を [NO_TITLE] 等に置換（参照実装と同じ）。
    """
    def _str(col: str, fill: str) -> pd.Series:
        if col not in df.columns:
            return pd.Series([fill if use_fill_map else ""] * len(df))
        # 先に str に変換（category のまま fillna すると「新しいカテゴリ」で TypeError になるため）
        s = df[col].astype(str).str.strip()
        if use_fill_map:
            s = s.replace("nan", fill)
        else:
            s = s.replace("nan", "")
        return s

    title = _str("movie_title", FILL_MAP["movie_title"])
    info = _str("movie_info", FILL_MAP["movie_info"])
    genres = _str("genres", FILL_MAP["genres"])
    directors = _str("directors", FILL_MAP["directors"])
    production = _str("production_company", FILL_MAP["production_company"])
    critic = _str("critic_name", FILL_MAP["critic_name"])
    top_critic = df["top_critic"].astype(str) if "top_critic" in df.columns else pd.Series(["False"] * len(df))
    publisher = _str("publisher_name", FILL_MAP["publisher_name"])
    content = _str("content_rating", FILL_MAP["content_rating"])

    parts = (
        "[MOVIE] " + title + " "
        + "[INFO] " + info + " "
        + "[GENRES] " + genres + " "
        + "[DIRECTORS] " + directors + " "
        + "[PRODUCTION] " + production + " "
        + "[CRITIC] " + critic + " "
        + "[TOP_CRITIC] " + top_critic + " "
        + "[PUBLISHER] " + publisher + " "
        + "[CONTENT] " + content
    )
    return parts.str.strip()


# デフォルト: ModernBERT（atmaCup 上位でよく使われる）。DeBERTa は "microsoft/deberta-v3-base" / "microsoft/deberta-v3-large"
DEFAULT_BERT_MODEL = "answerdotai/ModernBERT-base"


def run_bert_submission(
    train: pd.DataFrame,
    test: pd.DataFrame,
    submissions_dir: Path,
    *,
    model_name: str = DEFAULT_BERT_MODEL,
    n_folds: int = 2,
    max_length: int = 512,
    batch_size: int = 16,
    gradient_accumulation_steps: int = 2,
    epochs: int = 2,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    seed: int = 42,
    out_name: str = "submission_modernbert.csv",
    cv_strategy: str = "stratified",
    use_class_weight: bool = True,
    use_fill_map: bool = True,
    bf16: bool = True,
    cache_dir: Path | None = None,
    cache_name: str | None = None,
    log_fold: bool = True,
) -> dict[str, Any]:
    """
    BERT/ModernBERT/DeBERTa で 2-fold 学習し、テスト予測を平均して提出 CSV を保存する。

    cache_dir を指定すると各 fold のテスト予測を .npy で保存し、再実行時にその fold はスキップして再開できる。
    cache_name はキャッシュファイルのプレフィックス（省略時は out_name から自動）。
    log_fold=True で fold 完了時にログ出力。
    """
    try:
        import torch
        import torch.nn as nn
        # 古い PyTorch では torch.backends.mps.is_macos_or_newer が無いため、transformers の bf16 判定で落ちる。未定義なら追加し bf16 は使わない。
        if not hasattr(torch.backends.mps, "is_macos_or_newer"):
            torch.backends.mps.is_macos_or_newer = lambda _major, _minor: False
            bf16 = False
        from scipy.special import softmax
        from sklearn.model_selection import StratifiedKFold
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
            DataCollatorWithPadding,
        )
        from datasets import Dataset
    except ImportError as e:
        return {"ok": False, "path": submissions_dir / out_name, "message": f"transformers/torch 未導入: {e}"}

    y = train["target"].values.astype(np.float32)
    texts_train = build_bert_text(train, use_fill_map=use_fill_map)
    texts_test = build_bert_text(test, use_fill_map=use_fill_map)

    # CV 分割: stratified（参照実装） or time
    if cv_strategy == "stratified":
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        fold_splits = list(skf.split(np.arange(len(train)), train["target"].values))
        fold_indices = [np.arange(len(train))[tr_idx] for tr_idx, _ in fold_splits]
        val_indices = [np.arange(len(train))[val_idx] for _, val_idx in fold_splits]
    else:
        if "review_year" not in train.columns:
            return {"ok": False, "path": submissions_dir / out_name, "message": "train に review_year がありません"}
        years = sorted(train["review_year"].dropna().unique())
        n_folds = min(n_folds, len(years))
        val_years = list(years)[-n_folds:]
        fold_indices = [np.where((train["review_year"] < vy).values)[0] for vy in val_years]
        val_indices = [np.where((train["review_year"] == vy).values)[0] for vy in val_years]

    # クラス重み（参照実装: pos_weight = n_neg/n_pos）
    n_neg = int((y == 0).sum())
    n_pos = int((y == 1).sum())
    pos_weight = n_neg / max(n_pos, 1)
    class_weights = torch.tensor([1.0, pos_weight], dtype=torch.float32)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            w = class_weights.to(outputs.logits.device)
            loss = nn.CrossEntropyLoss(weight=w)(outputs.logits, labels)
            return (loss, outputs) if return_outputs else loss

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def tokenize_fn(batch, labels=None):
        enc = tokenizer(
            batch["text"],
            truncation=True,
            padding=False,
            max_length=max_length,
        )
        if labels is not None:
            enc["labels"] = labels
        return enc

    eval_steps = 500
    save_steps = 500
    all_test_preds = []
    _cache_name = cache_name if cache_name is not None else out_name.replace("submission_", "").replace(".csv", "")
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

    for fold in range(n_folds):
        tr_idx = fold_indices[fold]
        val_idx = val_indices[fold]
        if len(val_idx) == 0:
            continue

        cache_file = None
        if cache_dir is not None:
            cache_file = cache_dir / f"{_cache_name}_fold{fold}.npy"
            if cache_file.exists():
                p = np.load(cache_file)
                all_test_preds.append(p)
                if log_fold:
                    print(f"  [BERT cache] {_cache_name} fold {fold + 1}/{n_folds} を読み込み")
                continue

        if log_fold:
            print(f"  [BERT] {_cache_name} fold {fold + 1}/{n_folds} 学習中...")

        X_tr = texts_train.iloc[tr_idx].tolist()
        X_val = texts_train.iloc[val_idx].tolist()
        y_tr = y[tr_idx].round().astype(int).tolist()
        y_val = y[val_idx].round().astype(int).tolist()

        train_ds = Dataset.from_dict({"text": X_tr, "labels": y_tr})
        val_ds = Dataset.from_dict({"text": X_val, "labels": y_val})

        def _tok_train(batch):
            return tokenize_fn(batch, batch["labels"])

        train_ds = train_ds.map(_tok_train, batched=True, remove_columns=["text"])
        val_ds = val_ds.map(_tok_train, batched=True, remove_columns=["text"])

        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        fold_out = submissions_dir.parent / "bert_output" / f"{_cache_name}_fold{fold}"
        training_args = TrainingArguments(
            output_dir=str(fold_out),
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            lr_scheduler_type="cosine",
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=save_steps,
            load_best_model_at_end=True,
            metric_for_best_model="auc",
            greater_is_better=True,
            save_total_limit=1,
            bf16=bf16,
            seed=seed,
            report_to="none",
            dataloader_num_workers=0,  # macOS でマルチプロセス DataLoader が segfault するため
        )

        def compute_metrics(eval_pred):
            from sklearn.metrics import roc_auc_score
            logits, labels = eval_pred.predictions, eval_pred.label_ids
            probs = softmax(logits, axis=1)[:, 1]
            return {"auc": float(roc_auc_score(labels, probs))}

        trainer_cls = WeightedTrainer if use_class_weight else Trainer
        trainer = trainer_cls(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
        )
        trainer.train()

        # テスト予測（動的パディングのため tokenize のみ → collator が pad）
        test_ds = Dataset.from_dict({"text": texts_test.tolist()})
        test_ds = test_ds.map(
            lambda b: tokenize_fn(b),
            batched=True,
            remove_columns=["text"],
        )
        preds = trainer.predict(test_ds)
        p = np.clip(softmax(preds.predictions, axis=1)[:, 1], 0, 1).astype(np.float64)
        all_test_preds.append(p)
        if cache_dir is not None and cache_file is not None:
            np.save(cache_file, p)
        if log_fold:
            print(f"  [BERT] {_cache_name} fold {fold + 1}/{n_folds} 完了")

        del model, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not all_test_preds:
        return {"ok": False, "path": submissions_dir / out_name, "message": "有効な fold がありません"}

    pred_avg = np.mean(all_test_preds, axis=0)
    out_path = submissions_dir / out_name
    save_submission(test, pred_avg, out_path, sanitize=True)
    verified = verify_submission(out_path, test)
    return {"ok": verified.get("ok", True), "path": out_path, "message": verified.get("message", "OK")}


# 現時点の最高精度提出（0.76591: bpr64_count1 + BPR128 ブレンド）
DEFAULT_BEST_SUBMISSION = "submission_blend_bpr64_count1_bpr128.csv"


def blend_with_best_submission(
    submissions_dir: Path,
    test: pd.DataFrame,
    *,
    best_name: str = DEFAULT_BEST_SUBMISSION,
    bert_name: str = "submission_modernbert.csv",
    out_name: str = "submission_blend_best_bert.csv",
    weight_best: float = 0.5,
) -> dict[str, Any]:
    """
    現在の最高精度提出と BERT/ModernBERT の予測を加重平均し、1本の提出 CSV を保存する。

    weight_best=0.5 のとき 50% 最高精度 + 50% BERT。上げたい場合は weight_best を大きく（例 0.7）。
    """
    path_best = submissions_dir / best_name
    path_bert = submissions_dir / bert_name
    if not path_best.exists():
        return {"ok": False, "path": submissions_dir / out_name, "message": f"最高精度ファイルがありません: {best_name}"}
    if not path_bert.exists():
        return {"ok": False, "path": submissions_dir / out_name, "message": f"BERT 提出がありません: {bert_name}. 先に run_bert_submission を実行してください。"}

    df_best = pd.read_csv(path_best)[["ID", "target"]].rename(columns={"target": "best"})
    df_bert = pd.read_csv(path_bert)[["ID", "target"]].rename(columns={"target": "bert"})
    m = df_best.merge(df_bert, on="ID", how="inner")
    if len(m) != len(test):
        return {"ok": False, "path": submissions_dir / out_name, "message": f"ID 数が一致しません: best&bert の merge={len(m)}, test={len(test)}. 両 CSV は同じ test の提出である必要があります。"}
    m["target"] = (weight_best * m["best"] + (1 - weight_best) * m["bert"])
    m = m.set_index("ID").reindex(test["ID"]).reset_index()
    out_path = submissions_dir / out_name
    save_submission(test, m["target"].values, out_path, sanitize=True)
    verified = verify_submission(out_path, test)
    return {"ok": verified.get("ok", True), "path": out_path, "message": verified.get("message", "OK")}
