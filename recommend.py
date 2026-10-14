"""
recommend.py - the recommendation half of RikoNet (paper Sec. 3.1.2 - 3.1.5 and Fig. 1/2).

    1. build_embeddings()      three-stage autoencoder anime embeddings (Sec. 3.1.2)
    2. self_tuning_table()     candidate cluster counts for several neighbourhood sizes (Table 1)
    3. cluster_anime()         spectral clustering, nearest-neighbour affinity (Sec. 3.1.3)
    4. opposite_titles()       "logically opposite" title/cluster of every anime (Algorithm 1)
    5. hybrid_filter()         'Similar Anime' + 'Anime You May Like' lists (Sec. 3.1.5)

It is also a command-line tool once main.py has trained everything:
    python recommend.py                                  cold start (no ratings yet)
    python recommend.py --user 226                       an existing Kaggle user id
    python recommend.py --rate "Death Note=9" "Clannad=4" "Toradora!=8"
"""
import argparse
import html
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import kneighbors_graph

import config as C


# ============================================================================ 1. embeddings
def build_embeddings(data, X_train, X_val, seed=C.SEED, verbose=True):
    """Three autoencoders, exactly the paper's recipe (sizes are ours, the paper omits them):

    stage 1  anime vector = its ratings by the 'top raters' (users with > t training ratings)
             masked autoencoder  R^m -> R^d                         (rating-based knowledge)
    stage 2  append the one-hot genre vector:  R^d -> R^(d+g)         (content knowledge)
    stage 3  plain (unmasked, linear-output) autoencoder R^(d+g) -> R^d  = final hybrid embedding
    Only TRAINING ratings are used, so no test information leaks into the clusters.
    """
    from model import DeepAutoencoder

    counts = np.diff(X_train.indptr)
    top = np.where(counts > C.TOP_USER_MIN_RATINGS)[0]
    A_train = X_train[top].T.tocsr()          # items x m
    A_val = X_val[top].T.tocsr()
    if verbose:
        print(f"[embed] top raters m = {len(top)} (users with > {C.TOP_USER_MIN_RATINGS} ratings)")

    ae1 = DeepAutoencoder(len(top), hidden=C.EMB_HIDDEN, dropout=0.2, refeed=False, seed=seed)
    h1 = ae1.fit(A_train, A_val, verbose=verbose, label="embed-1", patience=10)
    code = ae1.encode(A_train)
    code = (code - code.mean(0)) / (code.std(0) + 1e-6)   # put code on the same scale as genres

    V = np.hstack([code, data.item_genres]).astype(np.float32)  # d' = d + g
    ae3 = DeepAutoencoder(V.shape[1], hidden=C.HYBRID_HIDDEN, dropout=0.0, final_activation=None,
                          masked=False, refeed=False, seed=seed)
    h3 = ae3.fit(V, None, epochs=150, verbose=verbose, label="embed-3")
    emb = ae3.encode(V)
    return emb, dict(top_users=int(len(top)), stage1=h1, stage3=h3)


# ============================================================================ 2. Table 1
def self_tuning_table(emb, neighbour_range=C.SELF_TUNING_RANGE, max_clusters=900, top=3):
    """For each neighbourhood size K, rank cluster counts c by the eigengap of the normalised
    nearest-neighbour affinity (largest gaps = most 'natural' numbers of clusters).

    NOTE: Zelnik-Manor & Perona's full method scores each c with an eigenvector-rotation cost; we use
    the simpler, widely used eigengap criterion from the same paper's discussion. This is an
    approximation and is labelled as such in the report.
    """
    out = {}
    for K in neighbour_range:
        A = kneighbors_graph(emb, n_neighbors=K, include_self=False)
        A = 0.5 * (A + A.T)
        d = np.asarray(A.sum(1)).ravel()
        Dm = sp.diags(1 / np.sqrt(d))
        Nrm = (Dm @ A @ Dm).toarray()
        ev = np.sort(np.linalg.eigvalsh(Nrm))[::-1]          # 1 >= ev_1 >= ev_2 ...
        gaps = ev[:max_clusters] - ev[1:max_clusters + 1]     # gap after c = index + 1 clusters
        gaps[0] = -np.inf                                     # ignore the trivial c = 1
        best = np.argsort(-gaps)[:top] + 1
        out[K] = list(map(int, best)) + [int(connected_components(A)[0])]
        print(f"[table1] K={K:2d}  best c = {out[K][:top]}   connected components = {out[K][-1]}")
    rows = [f"rank {r + 1}" for r in range(top)] + ["connected components"]
    return pd.DataFrame(out, index=rows)


# ============================================================================ 3. clustering
def cluster_anime(emb, n_clusters=C.N_CLUSTERS, n_neighbors=C.MIN_CLUSTER_ELEMS, seed=C.SEED):
    from sklearn.cluster import SpectralClustering
    sc = SpectralClustering(n_clusters=n_clusters, affinity="nearest_neighbors",
                            n_neighbors=n_neighbors, assign_labels="cluster_qr", random_state=seed)
    return sc.fit_predict(emb)


# ============================================================================ 4. Algorithm 1
def opposite_titles(emb, eps=C.EPSILON, chunk=512):
    """Paper Algorithm 1, vectorised. For every title a0:
        M        = centroid of all titles
        d-metric = d_i / (d0 + eps), inverted if > 1, then 1 - value      (0 = same distance from M)
        c-metric = cos(angle between M->a_i and M->a0) + 1                (0 = opposite direction)
        opposite title = argmin(d-metric + c-metric)
    """
    V = emb - emb.mean(0)
    dist = np.linalg.norm(V, axis=1)
    Vn = V / (dist[:, None] + eps)
    n = len(emb)
    opp = np.empty(n, dtype=int)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        dm = dist[None, :] / (dist[s:e, None] + eps)
        dm = np.where(dm > 1, 1 / np.maximum(dm, eps), dm)
        score = (1 - dm) + (Vn[s:e] @ Vn.T + 1)
        score[np.arange(e - s), np.arange(s, e)] = np.inf      # a title is not its own opposite
        opp[s:e] = score.argmin(1)
    return opp


# ============================================================================ 5. hybrid filter
def hybrid_filter(pred, rated_items, rated_values, labels, opp_cluster, k=C.TOP_K,
                  like_threshold=C.LIKE_THRESHOLD_ANIME, last_n=C.LAST_N_RATED):
    """pred         predicted ratings for ALL anime (from the primary autoencoder)
    rated_items  anime indices the user rated, in temporal order (oldest first)
    S = unrated anime;  C = union of clusters of the last 3 rated titles, using the title's own
    cluster if it was liked (rating >= threshold) else its logically opposite cluster.
    Similar Anime      = top-k of S intersect C by predicted rating
    Anime You May Like = top-k of S minus C by predicted rating
    """
    unrated = np.ones(len(pred), bool)
    unrated[rated_items] = False
    chosen = set()
    for it, r in zip(rated_items[-last_n:], rated_values[-last_n:]):
        chosen.add(labels[it] if r >= like_threshold else opp_cluster[it])
    in_c = np.isin(labels, list(chosen))
    score = np.where(unrated, pred, -np.inf)
    sim = np.where(in_c, score, -np.inf)
    may = np.where(~in_c, score, -np.inf)
    top = lambda s: [i for i in np.argsort(-s)[:k] if np.isfinite(s[i])]
    return top(sim), top(may), sorted(chosen)


def cold_start(names, genres, genre_names, members, k=C.TOP_K):
    """Empty profile: most popular titles of the most common genres (paper Sec. 3.2)."""
    out = []
    for g in np.argsort(-genres.sum(0))[:k]:
        idx = np.where(genres[:, g] > 0)[0]
        best = idx[np.argsort(-members[idx])]
        pick = next(i for i in best if i not in out)
        out.append(pick)
    return out


# ============================================================================ CLI
def _load_runtime():
    from model import DeepAutoencoder
    import json
    kb = np.load(C.MODEL_DIR / "anime_kb.npz", allow_pickle=True)
    cfg = json.loads((C.MODEL_DIR / "anime_ae_config.json").read_text())
    ae = DeepAutoencoder(len(kb["names"]), hidden=tuple(cfg["hidden"]), dropout=cfg["dropout"],
                         refeed=cfg["refeed"], l2=cfg["l2"])
    ae.model.load_weights(C.MODEL_DIR / "anime_primary_ae.weights.h5")
    return kb, ae


def _show(title, idx, kb, pred):
    names = [html.unescape(n) for n in kb["names"][idx]]
    df = pd.DataFrame({"anime": names, "predicted rating": np.round(pred[idx], 2),
                       "cluster": kb["labels"][idx]})
    df.index = range(1, len(df) + 1)
    print(f"\n=== {title} ===\n{df.to_string()}")
    return df


def main():
    ap = argparse.ArgumentParser(description="RikoNet anime recommendations")
    ap.add_argument("--user", type=int, help="Kaggle user_id (uses all of that user's ratings)")
    ap.add_argument("--rate", nargs="*", help='ratings in temporal order, e.g. "Death Note=9"')
    args = ap.parse_args()
    kb, ae = _load_runtime()
    names = np.array([html.unescape(n).casefold() for n in kb["names"]])

    if args.user is not None:
        r = pd.read_csv(C.DATA_DIR / "rating.csv")
        r = r[(r.user_id == args.user) & (r.rating > 0)]
        pos = {a: i for i, a in enumerate(kb["item_ids"])}
        r = r[r.anime_id.isin(pos)]
        items, values = r.anime_id.map(pos).values, r.rating.values.astype(float)
    elif args.rate:
        items, values = [], []
        for pair in args.rate:
            title, val = pair.rsplit("=", 1)
            hits = np.where(names == title.casefold())[0]
            if not len(hits):
                hits = np.where(np.char.find(names, title.casefold()) >= 0)[0]
            if not len(hits):
                print(f"! '{title}' not found - skipped")
                continue
            items.append(int(hits[0])); values.append(float(val))
            print(f"  matched '{title}' -> {kb['names'][hits[0]]}  ({val})")
        items, values = np.array(items, int), np.array(values)
    else:
        items, values = np.array([], int), np.array([])

    if len(items) == 0:
        idx = cold_start(kb["names"], kb["genres"], list(kb["genre_names"]), kb["members"])
        _show("Cold start: popular titles from the most common genres", idx, kb,
              np.zeros(len(kb["names"])))
        return
    x = np.zeros((1, len(kb["names"])), np.float32)
    x[0, items] = values
    pred = np.clip(ae.predict(x)[0], 1, 10)
    sim, may, chosen = hybrid_filter(pred, items, values, kb["labels"], kb["opp_cluster"])
    print(f"\nUser profile: {len(items)} ratings; clusters used for 'Similar': {chosen}")
    a = _show("Similar Anime", sim, kb, pred)
    b = _show("Anime You May Like", may, kb, pred)
    out = C.OUT_DIR / "last_recommendation.csv"
    pd.concat([a.assign(list="Similar Anime"), b.assign(list="Anime You May Like")]).to_csv(out)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
