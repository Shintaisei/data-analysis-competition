"""
Embedding 用の次元削減ラッパー。コンペでよく使う 8 パターンを統一 API で提供。
各手法は train のみで fit し、train / test を transform（リーク防止）。
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD, FastICA, NMF, KernelPCA, SparsePCA
from sklearn.random_projection import GaussianRandomProjection

# 利用可能な手法一覧（キー = 提出ファイル名などに使う識別子）
REDUCTION_METHODS = [
    "pca",              # 1. PCA（分散最大化）
    "truncated_svd",    # 2. TruncatedSVD（SVD、中心化なし）
    "ica",              # 3. ICA（独立成分）
    "random_projection",# 4. Gaussian Random Projection（ランダム射影）
    "sparse_pca",       # 5. Sparse PCA（スパース成分）
    "nmf",              # 6. NMF（非負値、embedding は min-max で非負化）
    "kernel_pca",      # 7. Kernel PCA（RBF カーネル、非線形）
    "umap",             # 8. UMAP（umap-learn 要）
]


def _make_non_negative(E: np.ndarray, E_min: float | None = None, E_max: float | None = None) -> np.ndarray:
    """NMF 用に embedding を非負にする（min-max で [0, 1]）。E_min/E_max を渡すと test を train と同じスケールに。"""
    if E_min is None:
        E_min, E_max = E.min(), E.max()
    if E_max - E_min < 1e-9:
        return np.zeros_like(E)
    return np.clip((E - E_min) / (E_max - E_min), 0.0, 1.0).astype(np.float32)


def fit_transform_embedding(
    E_train: np.ndarray,
    E_test: np.ndarray,
    method: str,
    n_components: int,
    random_state: int = 42,
):
    """
    E_train で fit し、E_train と E_test を transform して返す。
    戻り値: (train_reduced, test_reduced), 列プレフィックス用の名前
    """
    n_comp = min(n_components, E_train.shape[1])
    prefix = f"emb_{method}_{n_comp}"

    if method == "pca":
        reducer = PCA(n_components=n_comp, random_state=random_state)
        reducer.fit(E_train)
        train_t = reducer.transform(E_train)
        test_t = reducer.transform(E_test)

    elif method == "truncated_svd":
        reducer = TruncatedSVD(n_components=n_comp, random_state=random_state)
        reducer.fit(E_train)
        train_t = reducer.transform(E_train)
        test_t = reducer.transform(E_test)

    elif method == "ica":
        reducer = FastICA(n_components=n_comp, random_state=random_state, max_iter=500)
        train_t = reducer.fit_transform(E_train)
        test_t = reducer.transform(E_test)

    elif method == "random_projection":
        reducer = GaussianRandomProjection(n_components=n_comp, random_state=random_state)
        reducer.fit(E_train)
        train_t = reducer.transform(E_train)
        test_t = reducer.transform(E_test)

    elif method == "sparse_pca":
        reducer = SparsePCA(n_components=n_comp, random_state=random_state, max_iter=200)
        reducer.fit(E_train)
        train_t = reducer.transform(E_train)
        test_t = reducer.transform(E_test)

    elif method == "nmf":
        e_min, e_max = float(E_train.min()), float(E_train.max())
        E_train_nn = _make_non_negative(E_train, e_min, e_max)
        E_test_nn = _make_non_negative(E_test, e_min, e_max)
        reducer = NMF(n_components=n_comp, random_state=random_state, max_iter=400)
        train_t = reducer.fit_transform(E_train_nn)
        test_t = reducer.transform(E_test_nn)

    elif method == "kernel_pca":
        reducer = KernelPCA(
            n_components=n_comp,
            kernel="rbf",
            gamma=1.0 / E_train.shape[1],
            random_state=random_state,
        )
        train_t = reducer.fit_transform(E_train)
        test_t = reducer.transform(E_test)

    elif method == "umap":
        try:
            import umap
        except ImportError:
            raise ImportError(
                "UMAP を使うには pip install umap-learn が必要です。"
            )
        reducer = umap.UMAP(
            n_components=n_comp,
            random_state=random_state,
            metric="euclidean",
            n_neighbors=15,
            min_dist=0.1,
        )
        reducer.fit(E_train)
        train_t = reducer.transform(E_train)
        test_t = reducer.transform(E_test)

    else:
        raise ValueError(
            f"不明な手法: {method}. 利用可能: {REDUCTION_METHODS}"
        )

    # NaN/Inf が出た場合（UMAP や Kernel PCA の境界例など）は train の列中央値で埋める
    train_t = np.asarray(train_t, dtype=np.float32)
    test_t = np.asarray(test_t, dtype=np.float32)
    for j in range(train_t.shape[1]):
        col_tr = train_t[:, j]
        if not np.isfinite(col_tr).all():
            fill = np.nanmedian(col_tr)
            if not np.isfinite(fill):
                fill = 0.0
            train_t[:, j] = np.where(np.isfinite(col_tr), col_tr, fill)
        col_te = test_t[:, j]
        if not np.isfinite(col_te).all():
            fill = np.nanmedian(train_t[:, j])  # test は train の統計で埋める
            if not np.isfinite(fill):
                fill = 0.0
            test_t[:, j] = np.where(np.isfinite(col_te), col_te, fill)

    return train_t, test_t, prefix
