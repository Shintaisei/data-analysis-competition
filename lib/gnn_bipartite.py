"""
二部グラフ GNN（批評家–映画）で (critic, movie) の Fresh/Rotten を予測する。

1位解法風: SAGEConv + LayerNorm、接続情報のみで学習。train+test の全エッジでグラフを構築し、
train エッジで BCE 学習、test エッジで予測を出す。
学習後にモデルを保存し、同じデータなら保存済みモデルを読み込んで予測だけ行える。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from .submission import save_submission, verify_submission

# モデル保存用のデフォルトディレクトリ（outputs/submissions の兄弟）
DEFAULT_MODEL_DIR = Path("outputs") / "models"
DEFAULT_MODEL_FILENAME = "gnn_bipartite.pt"


class BipartiteGraphData(NamedTuple):
    """二部グラフと学習・予測用インデックス。"""
    n_critics: int
    n_movies: int
    edge_index: np.ndarray  # [2, E] int64, 無向なので 2*ペア数
    train_c_node: np.ndarray  # train 行の批評家ノード ID
    train_m_node: np.ndarray  # train 行の映画ノード ID
    train_y: np.ndarray       # train の target (0/1)
    test_c_node: np.ndarray   # test 行の批評家ノード ID（ctx.test の行順）
    test_m_node: np.ndarray   # test 行の映画ノード ID


def build_bipartite_graph(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> BipartiteGraphData:
    """
    train / test から批評家–映画の二部グラフを構築する。
    ノード: 0..n_c-1 = 批評家, n_c..n_c+n_m-1 = 映画。
    エッジ: train+test の全 (critic, movie) を無向で追加。
    """
    critics = pd.concat([
        train_df["critic_name"].astype(str),
        test_df["critic_name"].astype(str),
    ]).unique()
    movies = pd.concat([
        train_df["rotten_tomatoes_link"].astype(str),
        test_df["rotten_tomatoes_link"].astype(str),
    ]).unique()
    c2i = {c: i for i, c in enumerate(critics)}
    m2j = {m: j for j, m in enumerate(movies)}
    n_c, n_m = len(c2i), len(m2j)

    edges = []  # (c_node, m_node) のリスト。c_node in [0,n_c), m_node in [n_c, n_c+n_m)
    def add_pair(ci: int, mj: int) -> None:
        c_node = ci
        m_node = n_c + mj
        edges.append((c_node, m_node))
        edges.append((m_node, c_node))

    for _, r in train_df.iterrows():
        ci = c2i.get(str(r["critic_name"]))
        mj = m2j.get(str(r["rotten_tomatoes_link"]))
        if ci is not None and mj is not None:
            add_pair(ci, mj)
    for _, r in test_df.iterrows():
        ci = c2i.get(str(r["critic_name"]))
        mj = m2j.get(str(r["rotten_tomatoes_link"]))
        if ci is not None and mj is not None:
            add_pair(ci, mj)

    edge_index = np.array(edges, dtype=np.int64).T  # [2, E]

    train_c_node = np.array([
        c2i.get(str(r["critic_name"]), -1) for _, r in train_df.iterrows()
    ], dtype=np.int64)
    train_m_node = np.array([
        (n_c + m2j[str(r["rotten_tomatoes_link"])] if str(r["rotten_tomatoes_link"]) in m2j else -1)
        for _, r in train_df.iterrows()
    ], dtype=np.int64)
    train_y = train_df["target"].values.astype(np.float32)

    test_c_node = np.array([
        c2i.get(str(r["critic_name"]), -1) for _, r in test_df.iterrows()
    ], dtype=np.int64)
    test_m_node = np.array([
        (n_c + m2j[str(r["rotten_tomatoes_link"])] if str(r["rotten_tomatoes_link"]) in m2j else -1)
        for _, r in test_df.iterrows()
    ], dtype=np.int64)

    # train のみ不正ペアを除く。test は行順を保持し -1 は予測時に 0.5 にする
    train_ok = (train_c_node >= 0) & (train_m_node >= 0)
    train_c_node = train_c_node[train_ok]
    train_m_node = train_m_node[train_ok]
    train_y = train_y[train_ok]

    return BipartiteGraphData(
        n_critics=n_c,
        n_movies=n_m,
        edge_index=edge_index,
        train_c_node=train_c_node,
        train_m_node=train_m_node,
        train_y=train_y,
        test_c_node=test_c_node,
        test_m_node=test_m_node,
    )


def _get_bipartite_sage_class():
    """BipartiteSAGE クラスを返す（torch / PyG はここで import）。"""
    import torch
    from torch import nn
    from torch_geometric.data import Data
    from torch_geometric.nn import SAGEConv

    class BipartiteSAGE(nn.Module):
        def __init__(self, num_nodes: int, hidden: int, num_layers: int):
            super().__init__()
            self.embed = nn.Embedding(num_nodes, hidden)
            self.convs = nn.ModuleList()
            self.lns = nn.ModuleList()
            for _ in range(num_layers):
                self.convs.append(SAGEConv(hidden, hidden))
                self.lns.append(nn.LayerNorm(hidden))
            self.reset_parameters()

        def reset_parameters(self) -> None:
            nn.init.xavier_uniform_(self.embed.weight)
            for c in self.convs:
                c.reset_parameters()
            for ln in self.lns:
                ln.reset_parameters()

        def forward(self, data: Data) -> torch.Tensor:
            x = self.embed.weight
            for conv, ln in zip(self.convs, self.lns):
                x = conv(x, data.edge_index)
                x = ln(x)
                x = x.relu()
            return x

    return BipartiteSAGE


def _get_gnn_model_and_train(
    data: BipartiteGraphData,
    hidden_dim: int = 64,
    num_layers: int = 2,
    lr: float = 1e-2,
    epochs: int = 200,
    seed: int = 42,
    verbose: bool = True,
    save_path: Path | None = None,
) -> np.ndarray:
    """
    PyTorch / PyG で GNN を学習し、test の (c_node, m_node) に対する予測確率を返す。
    save_path を指定すると学習後に state_dict と設定を保存する。
    """
    import torch
    from torch import nn
    from torch_geometric.data import Data

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    n_nodes = data.n_critics + data.n_movies
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    edge_index_t = torch.from_numpy(data.edge_index).long().to(device)
    pyg_data = Data(edge_index=edge_index_t, num_nodes=n_nodes).to(device)

    BipartiteSAGE = _get_bipartite_sage_class()
    model = BipartiteSAGE(n_nodes, hidden_dim, num_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    train_c = torch.from_numpy(data.train_c_node).long().to(device)
    train_m = torch.from_numpy(data.train_m_node).long().to(device)
    train_y_t = torch.from_numpy(data.train_y).float().unsqueeze(1).to(device)

    # 進捗を 5 epoch ごと（epochs が少ないときは 1 ごと）に表示
    log_interval = 5 if epochs >= 20 else (2 if epochs >= 5 else 1)
    final_loss = 0.0
    if verbose:
        print("  [GNN] 学習ループ開始（1 epoch 目は数分かかることがあります）...", flush=True)
    for epoch in range(epochs):
        opt.zero_grad()
        h = model(pyg_data)
        logits = (h[train_c] * h[train_m]).sum(dim=1, keepdim=True)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, train_y_t)
        loss.backward()
        opt.step()
        final_loss = loss.item()
        if verbose and ((epoch + 1) % log_interval == 0 or epoch == 0 or epoch == epochs - 1):
            print(f"  [GNN] epoch {epoch + 1:4d}/{epochs}  loss={final_loss:.4f}", flush=True)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": model.state_dict(),
            "n_nodes": n_nodes,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
        }, save_path)
        if verbose:
            print(f"  [GNN] モデル保存: {save_path}", flush=True)

    model.eval()
    n_test = len(data.test_c_node)
    pred = np.full(n_test, 0.5, dtype=np.float64)
    valid = (data.test_c_node >= 0) & (data.test_m_node >= 0)
    if valid.any():
        with torch.no_grad():
            h = model(pyg_data)
            test_c = torch.from_numpy(data.test_c_node[valid]).long().to(device)
            test_m = torch.from_numpy(data.test_m_node[valid]).long().to(device)
            logits = (h[test_c] * h[test_m]).sum(dim=1)
            pred[valid] = torch.sigmoid(logits).cpu().numpy().astype(np.float64)
    if verbose:
        print(f"  [GNN] 最終 loss={final_loss:.4f}", flush=True)
    return pred


def _load_gnn_and_predict(
    data: BipartiteGraphData,
    load_path: Path,
    verbose: bool = True,
) -> np.ndarray:
    """保存済みモデルを読み込み、test の予測だけ行う。"""
    import torch
    from torch_geometric.data import Data

    load_path = Path(load_path)
    if not load_path.exists():
        raise FileNotFoundError(f"モデルが見つかりません: {load_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(load_path, map_location=device, weights_only=True)
    n_nodes = ckpt["n_nodes"]
    hidden_dim = ckpt["hidden_dim"]
    num_layers = ckpt["num_layers"]

    if n_nodes != data.n_critics + data.n_movies:
        raise ValueError(
            f"モデルの n_nodes={n_nodes} とグラフのノード数 {data.n_critics + data.n_movies} が一致しません。"
            " データが変わっている可能性があります。"
        )

    edge_index_t = torch.from_numpy(data.edge_index).long().to(device)
    pyg_data = Data(edge_index=edge_index_t, num_nodes=n_nodes).to(device)

    BipartiteSAGE = _get_bipartite_sage_class()
    model = BipartiteSAGE(n_nodes, hidden_dim, num_layers).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    n_test = len(data.test_c_node)
    pred = np.full(n_test, 0.5, dtype=np.float64)
    valid = (data.test_c_node >= 0) & (data.test_m_node >= 0)
    if valid.any():
        with torch.no_grad():
            h = model(pyg_data)
            test_c = torch.from_numpy(data.test_c_node[valid]).long().to(device)
            test_m = torch.from_numpy(data.test_m_node[valid]).long().to(device)
            logits = (h[test_c] * h[test_m]).sum(dim=1)
            pred[valid] = torch.sigmoid(logits).cpu().numpy().astype(np.float64)

    if verbose:
        print(f"  [GNN] 保存済みモデルで予測: {load_path}", flush=True)
    return pred


def run_gnn_bipartite(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    submissions_dir: Path,
    *,
    hidden_dim: int = 64,
    num_layers: int = 2,
    lr: float = 1e-2,
    epochs: int = 200,
    seed: int = 42,
    out_name: str = "submission_gnn_bipartite.csv",
    verbose: bool = True,
    model_dir: Path | None = DEFAULT_MODEL_DIR,
) -> dict[str, Any]:
    """
    二部グラフ GNN で学習・予測し、提出 CSV を保存する。
    model_dir を指定すると、学習後にモデルを model_dir/gnn_bipartite.pt に保存する。
    次回以降、同じデータで model_dir に保存済みがあれば学習をスキップして予測だけ行う。
    model_dir=None のときは保存・読み込みを行わず毎回学習する。
    """
    def log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    log("=" * 60)
    log("GNN 二部グラフ 学習・予測")
    log("=" * 60)

    try:
        data = build_bipartite_graph(train_df, test_df)
    except Exception as e:
        log(f"エラー（グラフ構築）: {e}")
        return {"ok": False, "path": str(submissions_dir / out_name), "message": f"グラフ構築エラー: {e}"}

    n_edges = data.edge_index.shape[1] // 2
    log(f"  グラフ: 批評家={data.n_critics:,}, 映画={data.n_movies:,}, エッジ(ペア数)={n_edges:,}")
    log(f"  学習サンプル={len(data.train_y):,}, 予測対象(test行)={len(data.test_c_node):,}")

    model_path = Path(model_dir) / DEFAULT_MODEL_FILENAME if model_dir else None
    if model_path is not None and model_path.exists():
        log(f"  保存済みモデルを読み込み: {model_path}")
        log("-" * 60)
        try:
            pred = _load_gnn_and_predict(data, model_path, verbose=verbose)
        except Exception as e:
            log(f"エラー（モデル読み込み）: {e}")
            return {"ok": False, "path": str(submissions_dir / out_name), "message": f"読み込みエラー: {e}"}
    else:
        if model_path is not None:
            log(f"  モデル保存先: {model_path}")
        log(f"  学習開始: epochs={epochs}, hidden={hidden_dim}, lr={lr}")
        log("-" * 60)
        try:
            pred = _get_gnn_model_and_train(
                data,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                lr=lr,
                epochs=epochs,
                seed=seed,
                verbose=verbose,
                save_path=model_path,
            )
        except Exception as e:
            log(f"エラー（学習・予測）: {e}")
            return {"ok": False, "path": str(submissions_dir / out_name), "message": f"学習・予測エラー: {e}"}

    if len(pred) != len(test_df):
        log(f"エラー: 予測長 {len(pred)} != test 行数 {len(test_df)}")
        return {
            "ok": False,
            "path": str(submissions_dir / out_name),
            "message": f"予測長 {len(pred)} != test 行数 {len(test_df)}",
        }

    log("-" * 60)
    log(f"  予測: min={pred.min():.4f}, max={pred.max():.4f}, mean={pred.mean():.4f}")

    path = Path(submissions_dir) / out_name
    save_submission(test_df, pred, path, sanitize=True)
    result = verify_submission(path, test_df)

    if result["ok"]:
        log(f"  保存完了: {path}")
        log(f"  検証: OK (行数={result['rows']})")
    else:
        log(f"  検証: NG - {result['message']}")
    log("=" * 60)
    return result