"""
baselines.py - simple reference recommenders, trained on exactly the same training split as
RikoNet so that the comparison is fair (the paper instead copied baseline numbers from other
papers, which used different splits).

All models share one interface:
    fit(X_train)                    X_train = sparse user x item rating matrix (0 = unknown)
    predict_rows(rows) -> ndarray   predicted ratings for those users on ALL items
"""
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize


class GlobalMean:
    name = "Global average"

    def fit(self, X):
        self.mu = X.data.mean()
        self.n_items = X.shape[1]
        return self

    def predict_rows(self, rows):
        return np.full((len(rows), self.n_items), self.mu, dtype=np.float32)


class UserMean(GlobalMean):
    name = "User average"

    def fit(self, X):
        super().fit(X)
        cnt = np.diff(X.indptr)
        self.user_mean = np.where(cnt > 0, np.asarray(X.sum(1)).ravel() / np.maximum(cnt, 1), self.mu)
        return self

    def predict_rows(self, rows):
        return np.repeat(self.user_mean[rows, None], self.n_items, 1).astype(np.float32)


class ItemMean(GlobalMean):
    name = "Item average"

    def fit(self, X):
        super().fit(X)
        Xc = X.tocsc()
        cnt = np.diff(Xc.indptr)
        self.item_mean = np.where(cnt > 0, np.asarray(X.sum(0)).ravel() / np.maximum(cnt, 1), self.mu)
        return self

    def predict_rows(self, rows):
        return np.repeat(self.item_mean[None, :], len(rows), 0).astype(np.float32)


class BiasBaseline(GlobalMean):
    """r_ui = mu + b_u + b_i with L2-regularised biases (Koren 2008), fitted by alternating updates."""
    name = "Bias baseline (mu+b_u+b_i)"

    def __init__(self, reg_u=10.0, reg_i=10.0, iters=10):
        self.reg_u, self.reg_i, self.iters = reg_u, reg_i, iters

    def fit(self, X):
        super().fit(X)
        Xo = X.tocoo()
        u, i, r = Xo.row, Xo.col, Xo.data.astype(np.float64)
        nu, ni = X.shape
        self.bu, self.bi = np.zeros(nu), np.zeros(ni)
        for _ in range(self.iters):
            self.bi = np.bincount(i, r - self.mu - self.bu[u], ni) / (np.bincount(i, None, ni) + self.reg_i)
            self.bu = np.bincount(u, r - self.mu - self.bi[i], nu) / (np.bincount(u, None, nu) + self.reg_u)
        return self

    def residuals(self, X):
        """Sparse matrix of r_ui - (mu + b_u + b_i) on the observed entries."""
        Xo = X.tocoo()
        e = Xo.data - (self.mu + self.bu[Xo.row] + self.bi[Xo.col])
        e[e == 0] = 1e-6   # keep explicit entries even if the residual is exactly 0
        return sp.csr_matrix((e, (Xo.row, Xo.col)), shape=X.shape)

    def predict_rows(self, rows):
        return (self.mu + self.bu[rows, None] + self.bi[None, :]).astype(np.float32)


class KNN(BiasBaseline):
    """Neighbourhood CF on baseline residuals (cosine similarity, top-k neighbours).

    item-based: r_ui = b_ui + sum_j s_ij * e_uj / sum_j |s_ij|   over the k most similar items j rated by u
    user-based: r_ui = b_ui + sum_v s_uv * e_vi / sum_v |s_uv|   over the k most similar users v
    """

    def __init__(self, kind="item", k=40, chunk=1024):
        super().__init__()
        self.kind, self.k, self.chunk = kind, k, chunk
        self.name = f"{kind.capitalize()} KNN (cosine, k={k})"

    def _topk_similarity(self, M):
        """Row-wise cosine similarity of M, keeping only the k largest per row (sparse)."""
        Mn = normalize(M, axis=1)
        n = M.shape[0]
        rows, cols, vals = [], [], []
        for s in range(0, n, self.chunk):
            S = (Mn[s:s + self.chunk] @ Mn.T).toarray()
            S[np.arange(S.shape[0]), np.arange(s, s + S.shape[0])] = 0     # no self-similarity
            idx = np.argpartition(-S, self.k, axis=1)[:, :self.k]
            rows.append(np.repeat(np.arange(s, s + S.shape[0]), self.k))
            cols.append(idx.ravel())
            vals.append(np.take_along_axis(S, idx, 1).ravel())
        return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(n, n))

    def fit(self, X):
        super().fit(X)
        self.E = self.residuals(X)                       # users x items
        self.I = (X > 0).astype(np.float64).tocsr()      # rated indicator
        M = self.E.T.tocsr() if self.kind == "item" else self.E
        self.S = self._topk_similarity(M)
        self.Sabs = abs(self.S)
        return self

    def predict_rows(self, rows):
        base = super().predict_rows(rows).astype(np.float64)
        if self.kind == "item":
            num = (self.E[rows] @ self.S.T).toarray()
            den = (self.I[rows] @ self.Sabs.T).toarray()
        else:
            num = (self.S[rows] @ self.E).toarray()
            den = (self.Sabs[rows] @ self.I).toarray()
        adj = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-9)
        return (base + adj).astype(np.float32)


class BiasedMF(BiasBaseline):
    """Biased matrix factorisation r_ui = mu + b_u + b_i + p_u.q_i, fitted with ALS on residuals."""

    def __init__(self, factors=32, reg=10.0, iters=15, seed=42):
        super().__init__()
        self.f, self.reg, self.mf_iters, self.seed = factors, reg, iters, seed
        self.name = f"Biased MF (ALS, f={factors})"

    def fit(self, X):
        super().fit(X)
        E = self.residuals(X).tocsr()
        Ec = E.tocsc()
        rng = np.random.default_rng(self.seed)
        nu, ni = E.shape
        self.P = rng.normal(0, 0.1, (nu, self.f))
        self.Q = rng.normal(0, 0.1, (ni, self.f))
        eye = np.eye(self.f)
        for _ in range(self.mf_iters):
            for u in range(nu):
                idx = E.indices[E.indptr[u]:E.indptr[u + 1]]
                if len(idx):
                    Qi = self.Q[idx]
                    self.P[u] = np.linalg.solve(Qi.T @ Qi + self.reg * eye,
                                                Qi.T @ E.data[E.indptr[u]:E.indptr[u + 1]])
            for i in range(ni):
                idx = Ec.indices[Ec.indptr[i]:Ec.indptr[i + 1]]
                if len(idx):
                    Pu = self.P[idx]
                    self.Q[i] = np.linalg.solve(Pu.T @ Pu + self.reg * eye,
                                                Pu.T @ Ec.data[Ec.indptr[i]:Ec.indptr[i + 1]])
        return self

    def predict_rows(self, rows):
        return (super().predict_rows(rows) + self.P[rows] @ self.Q.T).astype(np.float32)


class Popularity(GlobalMean):
    """Ranking-only baseline: recommend the items with the most training ratings."""
    name = "Most popular"

    def fit(self, X):
        super().fit(X)
        self.pop = np.diff(X.tocsc().indptr).astype(np.float32)
        return self

    def predict_rows(self, rows):
        return np.repeat(self.pop[None, :], len(rows), 0)


class RandomRanker(GlobalMean):
    name = "Random"

    def predict_rows(self, rows):
        return np.random.default_rng(len(rows)).random((len(rows), self.n_items)).astype(np.float32)


def rating_baselines():
    return [GlobalMean(), UserMean(), ItemMean(), BiasBaseline(), KNN("user"), KNN("item"),
            BiasedMF()]
