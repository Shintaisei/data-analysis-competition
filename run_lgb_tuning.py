#!/usr/bin/env python3
"""
LightGBM Optuna チューニングをバックグラウンドで実行するスクリプト。
プロジェクトルートで実行すること: python run_lgb_tuning.py [N_TRIALS]

例（裏で回す）:
  nohup python run_lgb_tuning.py 80 > lgb_tuning.log 2>&1 &
  # または tmux/screen の中で: python run_lgb_tuning.py 80
"""
import os
import sys
import random
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

# スクリプトの場所 = プロジェクトルートに移動（どこから実行しても動くように）
_root = os.path.dirname(os.path.abspath(__file__))
os.chdir(_root)
if _root not in sys.path:
    sys.path.insert(0, _root)

from lib import get_baseline_data, BASELINE_FEATURES

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    sys.exit("Optuna が必要です: pip install optuna")

OUTPUT_DIR = "outputs"
EARLY_STOPPING_ROUNDS = 30
VAL_YEARS = [2013, 2014, 2015, 2016]


def log(msg):
    print(msg, flush=True)


def main():
    parser = argparse.ArgumentParser(description="LGB Optuna tuning (run from project root)")
    parser.add_argument("n_trials", nargs="?", type=int, default=50, help="Optuna trials (default: 50)")
    args = parser.parse_args()
    n_trials = args.n_trials

    def seed_everything(seed=42):
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)

    seed_everything(42)
    os.makedirs(os.path.join(OUTPUT_DIR, "submissions"), exist_ok=True)

    log("Loading data...")
    train, test = get_baseline_data()
    FEATURES = BASELINE_FEATURES

    time_splits = []
    for vy in VAL_YEARS:
        tr_idx = np.where(train["review_year"] < vy)[0]
        val_idx = np.where(train["review_year"] == vy)[0]
        if len(val_idx) > 0:
            time_splits.append((tr_idx, val_idx))

    X = train[FEATURES]
    y = train["target"].values
    X_test = test[FEATURES]
    log(f"Train: {len(train):,}, Test: {len(test):,}, Features: {len(FEATURES)}")
    log(f"時系列CV: {len(time_splits)} folds, N_TRIALS={n_trials}\n")

    def run_cv_with_params(params):
        fold_scores = []
        for tr_idx, val_idx in time_splits:
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]
            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
            )
            pred = model.predict_proba(X_val)[:, 1]
            fold_scores.append(roc_auc_score(y_val, pred))
        return np.mean(fold_scores)

    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "random_state": 42,
            "verbosity": -1,
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 200),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
        }
        return run_cv_with_params(params)

    log("Optuna tuning started...")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=15),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    log(f"\nBest CV AUC: {study.best_value:.4f}")
    log(f"Best params: {study.best_params}")

    lgb_params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "random_state": 42,
        "verbosity": -1,
        **study.best_params,
    }
    lgb_params["n_estimators"] = max(lgb_params.get("n_estimators", 500), 400)

    log("\nRe-running CV with best params...")
    for fold, (tr_idx, val_idx) in enumerate(time_splits):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        model = lgb.LGBMClassifier(**lgb_params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
        log(f"  Fold{fold+1}: val_year={VAL_YEARS[fold]}, AUC={auc:.4f}")

    log("\nTraining on full train and saving submission...")
    model_full = lgb.LGBMClassifier(**lgb_params)
    model_full.fit(X, y)
    final_pred = model_full.predict_proba(X_test)[:, 1]
    submission = pd.DataFrame({"ID": test["ID"], "target": final_pred})
    out_path = os.path.join(OUTPUT_DIR, "submissions", "submission_lgb_tuned.csv")
    submission.to_csv(out_path, index=False)
    log(f"Saved {out_path} (rows: {len(submission):,})")
    log("Done.")


if __name__ == "__main__":
    main()
