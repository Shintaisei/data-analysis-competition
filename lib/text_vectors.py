"""
テキスト（movie_info 等）を固定次元ベクトルに変換する手法をまとめる。
時系列 CV 用: tr のみで fit し、val/te には transform のみ適用する想定。

各 build_* は (mi_tr, mi_val, mi_te) を受け、
(tr_mat, val_mat, te_mat, prefix) を返す。
tr_mat は (n_tr, d) の ndarray。val_mat / te_mat は None または (n, d)。
prefix は列名の接頭辞（例: "tfidf" → tfidf_0, tfidf_1, ...）。
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD

# 共通パラメータ
TFIDF_KW = dict(min_df=2, max_df=0.95, ngram_range=(1, 2), sublinear_tf=True)
COUNT_KW = dict(min_df=2, max_df=0.95, ngram_range=(1, 2))


def _to_dense(X):
    if hasattr(X, "todense"):
        return np.asarray(X.todense())
    return np.asarray(X)


def build_tfidf(mi_tr, mi_val, mi_te, n_features=50, prefix="tfidf"):
    vec = TfidfVectorizer(max_features=n_features, **TFIDF_KW)
    tr_mat = _to_dense(vec.fit_transform(mi_tr))
    n_actual = tr_mat.shape[1]
    val_mat = _to_dense(vec.transform(mi_val))[:, :n_actual] if mi_val is not None else None
    te_mat = _to_dense(vec.transform(mi_te))[:, :n_actual] if mi_te is not None else None
    return tr_mat, val_mat, te_mat, prefix


def build_tfidf_svd(mi_tr, mi_val, mi_te, n_tfidf=200, n_components=20, prefix="tfidf_svd"):
    vec = TfidfVectorizer(max_features=n_tfidf, **TFIDF_KW)
    X_tr = vec.fit_transform(mi_tr)
    n_actual = min(n_components, X_tr.shape[1])
    svd = TruncatedSVD(n_components=n_actual, random_state=42)
    tr_mat = svd.fit_transform(X_tr)
    val_mat = svd.transform(vec.transform(mi_val)) if mi_val is not None else None
    te_mat = svd.transform(vec.transform(mi_te)) if mi_te is not None else None
    return tr_mat, val_mat, te_mat, prefix


def build_count(mi_tr, mi_val, mi_te, n_features=50, prefix="count"):
    vec = CountVectorizer(max_features=n_features, **COUNT_KW)
    tr_mat = _to_dense(vec.fit_transform(mi_tr))
    n_actual = tr_mat.shape[1]
    val_mat = _to_dense(vec.transform(mi_val))[:, :n_actual] if mi_val is not None else None
    te_mat = _to_dense(vec.transform(mi_te))[:, :n_actual] if mi_te is not None else None
    return tr_mat, val_mat, te_mat, prefix


def build_count_svd(mi_tr, mi_val, mi_te, n_count=200, n_components=20, prefix="count_svd"):
    vec = CountVectorizer(max_features=n_count, **COUNT_KW)
    X_tr = vec.fit_transform(mi_tr)
    n_actual = min(n_components, X_tr.shape[1])
    svd = TruncatedSVD(n_components=n_actual, random_state=42)
    tr_mat = svd.fit_transform(X_tr)
    val_mat = svd.transform(vec.transform(mi_val)) if mi_val is not None else None
    te_mat = svd.transform(vec.transform(mi_te)) if mi_te is not None else None
    return tr_mat, val_mat, te_mat, prefix


def build_hash(mi_tr, mi_val, mi_te, n_features=50, n_hash=256, prefix="hash"):
    vec = HashingVectorizer(n_features=n_hash, ngram_range=(1, 2), norm="l2")
    tr_mat = _to_dense(vec.transform(mi_tr))[:, :n_features]
    n_actual = tr_mat.shape[1]
    val_mat = _to_dense(vec.transform(mi_val))[:, :n_actual] if mi_val is not None else None
    te_mat = _to_dense(vec.transform(mi_te))[:, :n_actual] if mi_te is not None else None
    return tr_mat, val_mat, te_mat, prefix


def build_hash_svd(mi_tr, mi_val, mi_te, n_hash=512, n_components=20, prefix="hash_svd"):
    vec = HashingVectorizer(n_features=n_hash, ngram_range=(1, 2), norm="l2")
    X_tr = vec.transform(mi_tr)
    n_actual = min(n_components, X_tr.shape[1])
    svd = TruncatedSVD(n_components=n_actual, random_state=42)
    tr_mat = svd.fit_transform(X_tr)
    val_mat = svd.transform(vec.transform(mi_val)) if mi_val is not None else None
    te_mat = svd.transform(vec.transform(mi_te)) if mi_te is not None else None
    return tr_mat, val_mat, te_mat, prefix


def build_lda(mi_tr, mi_val, mi_te, n_components=20, n_tfidf=200, max_iter=10, prefix="lda"):
    vec = TfidfVectorizer(max_features=n_tfidf, **TFIDF_KW)
    X_tr = vec.fit_transform(mi_tr)
    n_actual = min(n_components, X_tr.shape[1])
    lda = LatentDirichletAllocation(n_components=n_actual, random_state=42, max_iter=max_iter)
    tr_mat = lda.fit_transform(X_tr)
    val_mat = lda.transform(vec.transform(mi_val)) if mi_val is not None else None
    te_mat = lda.transform(vec.transform(mi_te)) if mi_te is not None else None
    return tr_mat, val_mat, te_mat, prefix


def build_nmf(mi_tr, mi_val, mi_te, n_components=20, n_tfidf=200, max_iter=100, prefix="nmf"):
    vec = TfidfVectorizer(max_features=n_tfidf, **TFIDF_KW)
    X_tr = vec.fit_transform(mi_tr)
    n_actual = min(n_components, X_tr.shape[1])
    nmf = NMF(n_components=n_actual, random_state=42, max_iter=max_iter)
    tr_mat = nmf.fit_transform(X_tr)
    val_mat = nmf.transform(vec.transform(mi_val)) if mi_val is not None else None
    te_mat = nmf.transform(vec.transform(mi_te)) if mi_te is not None else None
    return tr_mat, val_mat, te_mat, prefix


def build_doc2vec(mi_tr, mi_val, mi_te, vector_size=32, window=5, min_count=2, epochs=5, seed=42, prefix="d2v"):
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument
    docs = [TaggedDocument(words=str(t).split(), tags=[i]) for i, t in enumerate(mi_tr)]
    model = Doc2Vec(vector_size=vector_size, window=window, min_count=min_count, epochs=epochs, seed=seed, workers=1)
    model.build_vocab(docs)
    model.train(docs, total_examples=model.corpus_count, epochs=model.epochs)
    tr_mat = np.array([model.dv[i] for i in range(len(mi_tr))])
    val_mat = np.array([model.infer_vector(str(t).split()) for t in mi_val]) if mi_val is not None else None
    te_mat = np.array([model.infer_vector(str(t).split()) for t in mi_te]) if mi_te is not None else None
    return tr_mat, val_mat, te_mat, prefix


def build_word2vec_avg(mi_tr, mi_val, mi_te, vector_size=32, window=5, min_count=2, epochs=5, seed=42, prefix="w2v"):
    from gensim.models import Word2Vec
    sentences = [str(t).split() for t in mi_tr]
    model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count, epochs=epochs, seed=seed, workers=1)
    def doc_vec(words):
        vecs = [model.wv[w] for w in words if w in model.wv]
        return np.mean(vecs, axis=0) if vecs else np.zeros(vector_size)
    tr_mat = np.array([doc_vec(str(t).split()) for t in mi_tr])
    val_mat = np.array([doc_vec(str(t).split()) for t in mi_val]) if mi_val is not None else None
    te_mat = np.array([doc_vec(str(t).split()) for t in mi_te]) if mi_te is not None else None
    return tr_mat, val_mat, te_mat, prefix


def build_sentence_transformer(emb_tr, emb_val, emb_te, n_dims=32, prefix="st"):
    """事前計算済み埋め込みをそのまま使う。emb_* は (n, D) の ndarray。先頭 n_dims を使う。"""
    tr_mat = np.asarray(emb_tr)[:, :n_dims]
    val_mat = np.asarray(emb_val)[:, :n_dims] if emb_val is not None else None
    te_mat = np.asarray(emb_te)[:, :n_dims] if emb_te is not None else None
    return tr_mat, val_mat, te_mat, prefix


# config_name -> (builder_fn, needs_mi, needs_st_emb)
REGISTRY = {
    "base": None,
    "tfidf_50": (lambda mi_tr, mi_val, mi_te: build_tfidf(mi_tr, mi_val, mi_te, 50), True, False),
    "tfidf_100": (lambda mi_tr, mi_val, mi_te: build_tfidf(mi_tr, mi_val, mi_te, 100), True, False),
    "tfidf_svd20": (lambda mi_tr, mi_val, mi_te: build_tfidf_svd(mi_tr, mi_val, mi_te, 200, 20), True, False),
    "count_50": (lambda mi_tr, mi_val, mi_te: build_count(mi_tr, mi_val, mi_te, 50), True, False),
    "count_svd20": (lambda mi_tr, mi_val, mi_te: build_count_svd(mi_tr, mi_val, mi_te, 200, 20), True, False),
    "hash_50": (lambda mi_tr, mi_val, mi_te: build_hash(mi_tr, mi_val, mi_te, 50), True, False),
    "hash_svd20": (lambda mi_tr, mi_val, mi_te: build_hash_svd(mi_tr, mi_val, mi_te, 512, 20), True, False),
    "lda_10": (lambda mi_tr, mi_val, mi_te: build_lda(mi_tr, mi_val, mi_te, 10), True, False),
    "lda_20": (lambda mi_tr, mi_val, mi_te: build_lda(mi_tr, mi_val, mi_te, 20), True, False),
    "nmf_10": (lambda mi_tr, mi_val, mi_te: build_nmf(mi_tr, mi_val, mi_te, 10), True, False),
    "nmf_20": (lambda mi_tr, mi_val, mi_te: build_nmf(mi_tr, mi_val, mi_te, 20), True, False),
    "doc2vec_32": (lambda mi_tr, mi_val, mi_te: build_doc2vec(mi_tr, mi_val, mi_te, 32), True, False),
    "word2vec_32": (lambda mi_tr, mi_val, mi_te: build_word2vec_avg(mi_tr, mi_val, mi_te, 32), True, False),
    "sentence_transformer_32": None,  # 特別扱い: st_emb を渡す
    "concat_mi_title_tfidf_50": (lambda a, b, c: build_tfidf(a, b, c, 50, "ct_tfidf"), True, False),
    "concat_mi_title_tfidf_svd20": (lambda a, b, c: build_tfidf_svd(a, b, c, 200, 20, "ct_tfidf_svd"), True, False),
    "concat_mi_title_count_50": (lambda a, b, c: build_count(a, b, c, 50, "ct_count"), True, False),
}


def build_vectors(config_name, mi_tr, mi_val, mi_te, st_emb_tr=None, st_emb_val=None, st_emb_te=None):
    """
    config_name に応じてベクトル化し (tr_mat, val_mat, te_mat, prefix) を返す。
    base のときは (None, None, None, None)。
    sentence_transformer_32 のときは st_emb_* を渡すこと。
    失敗時（ImportError 等）は (None, None, None, None)。
    """
    if config_name == "base":
        return None, None, None, None
    if config_name == "sentence_transformer_32":
        if st_emb_tr is None or st_emb_te is None:
            return None, None, None, None
        return build_sentence_transformer(st_emb_tr, st_emb_val, st_emb_te, 32, "st")
    entry = REGISTRY.get(config_name)
    if entry is None:
        return None, None, None, None
    fn = entry[0]
    try:
        return fn(mi_tr, mi_val, mi_te)
    except Exception:
        return None, None, None, None


def get_available_configs(include_doc2vec=True, include_word2vec=True, include_sentence_transformer=False):
    """利用可能な config のリスト。オプション依存で doc2vec / word2vec / sentence_transformer を含むか決める。"""
    base_list = [
        "base", "tfidf_50", "tfidf_100", "tfidf_svd20",
        "count_50", "count_svd20", "hash_50", "hash_svd20",
        "lda_10", "lda_20", "nmf_10", "nmf_20",
        "concat_mi_title_tfidf_50", "concat_mi_title_tfidf_svd20", "concat_mi_title_count_50",
    ]
    if include_doc2vec:
        try:
            from gensim.models.doc2vec import Doc2Vec
            base_list.append("doc2vec_32")
        except ImportError:
            pass
    if include_word2vec:
        try:
            from gensim.models import Word2Vec
            base_list.append("word2vec_32")
        except ImportError:
            pass
    if include_sentence_transformer:
        base_list.append("sentence_transformer_32")
    return base_list


def get_config_descriptions():
    """各 config の短い説明（Markdown 表用）。"""
    return {
        "base": "なし（34 特徴のみ）",
        "tfidf_50": "TF-IDF 50 次元",
        "tfidf_100": "TF-IDF 100 次元",
        "tfidf_svd20": "TF-IDF(200) → SVD 20 次元",
        "count_50": "CountVectorizer 50 次元（BoW）",
        "count_svd20": "Count(200) → SVD 20 次元",
        "hash_50": "HashingVectorizer 先頭 50 次元",
        "hash_svd20": "Hash(512) → SVD 20 次元",
        "lda_10": "LDA 10 トピック",
        "lda_20": "LDA 20 トピック",
        "nmf_10": "NMF 10 次元",
        "nmf_20": "NMF 20 次元",
        "doc2vec_32": "Doc2Vec 32 次元（gensim）",
        "word2vec_32": "Word2Vec 平均 32 次元（gensim）",
        "sentence_transformer_32": "sentence-transformers 32 次元（事前計算）",
        "concat_mi_title_tfidf_50": "タイトル＋あらすじ結合 → TF-IDF 50 次元",
        "concat_mi_title_tfidf_svd20": "タイトル＋あらすじ結合 → TF-IDF(200)→SVD 20 次元",
        "concat_mi_title_count_50": "タイトル＋あらすじ結合 → Count 50 次元",
    }
