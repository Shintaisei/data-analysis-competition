#!/usr/bin/env python3
"""GNN 二部グラフを学習・予測し、提出 CSV を保存する。"""
import warnings
warnings.filterwarnings("ignore")

from lib.improvement_candidates import get_setup, run_gnn_bipartite_submission

if __name__ == "__main__":
    ctx = get_setup(seed=42, output_dir="outputs", use_best_pipeline=True)
    r = run_gnn_bipartite_submission(ctx, epochs=200, out_name="submission_gnn_bipartite.csv")
    print("OK" if r.get("ok") else r.get("message"))
