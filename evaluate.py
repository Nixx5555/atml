"""
evaluate.py - metrics and figures.

Rating-prediction metrics (paper eq. 4-5, computed only on held-out ratings):
    MSE  = mean (r - y)^2      RMSE = sqrt(MSE)      MAE = mean |r - y|
Top-K recommendation metrics (added by us; the paper only reports MSE/RMSE + a user study):
    Precision@K = hits in top-K / K
    Recall@K    = hits in top-K / number of relevant held-out items
    NDCG@K      = position-discounted hits, normalised by the best possible ordering
    HitRate@K   = share of users with at least one hit
  A "relevant" item = a held-out (test) rating >= threshold. Items the user already rated in
  train/validation are removed from the candidate list, so the model cannot score easy wins.
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.sparse as sp

import config as C

# ---- palette (validated colour-blind-safe reference palette, light mode) -------------------
BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"
GREY, INK, INK2, GRID = "#a9a8a2", "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.bbox": "tight", "font.size": 10, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
    "axes.titlesize": 11, "axes.titlecolor": INK, "legend.frameon": False,
})


# ============================================================================ metrics
def rating_metrics(predict_rows, X_test, r_min, r_max, batch=1024):
    """predict_rows(rows) -> predicted ratings for those users on all items."""
    X_test = sp.csr_matrix(X_test)
    users = np.unique(X_test.nonzero()[0])
    errs = []
    for s in range(0, len(users), batch):
        rows = users[s:s + batch]
        P = np.clip(predict_rows(rows), r_min, r_max)
        T = X_test[rows].tocoo()
        errs.append(P[T.row, T.col] - T.data)
    e = np.concatenate(errs)
    mse = float((e ** 2).mean())
    return dict(MSE=mse, RMSE=float(np.sqrt(mse)), MAE=float(np.abs(e).mean()), n=int(len(e)))


def ranking_metrics(score_rows, X_known, X_test, rel_threshold, k=C.TOP_K, batch=1024):
    """score_rows(rows) -> a score per item (higher = recommend first)."""
    X_known, X_test = sp.csr_matrix(X_known), sp.csr_matrix(X_test)
    rel = X_test.multiply(X_test >= rel_threshold).tocsr()
    users = np.where(np.diff(rel.indptr) > 0)[0]
    disc = 1 / np.log2(np.arange(2, k + 2))
    P, R, N, H, shown = [], [], [], [], set()
    for s in range(0, len(users), batch):
        rows = users[s:s + batch]
        S = score_rows(rows).astype(np.float64)
        S[X_known[rows].nonzero()] = -np.inf               # never recommend already-rated items
        top = np.argpartition(-S, k, axis=1)[:, :k]
        top = np.take_along_axis(top, np.argsort(-np.take_along_axis(S, top, 1), 1), 1)
        Rel = rel[rows]
        for j in range(len(rows)):
            relset = set(Rel.indices[Rel.indptr[j]:Rel.indptr[j + 1]])
            hits = np.array([t in relset for t in top[j]], float)
            P.append(hits.sum() / k)
            R.append(hits.sum() / len(relset))
            N.append((hits * disc).sum() / disc[:min(len(relset), k)].sum())
            H.append(float(hits.any()))
            shown.update(top[j].tolist())
    return {f"Precision@{k}": float(np.mean(P)), f"Recall@{k}": float(np.mean(R)),
            f"NDCG@{k}": float(np.mean(N)), f"HitRate@{k}": float(np.mean(H)),
            "Coverage": len(shown) / X_known.shape[1], "users": int(len(users))}


def list_precision(lists, rel_sets, k=C.TOP_K):
    """Precision/HitRate of arbitrary per-user recommendation lists (used for the hybrid filter)."""
    p = [len(set(l) & r) / k for l, r in zip(lists, rel_sets)]
    h = [float(len(set(l) & r) > 0) for l, r in zip(lists, rel_sets)]
    rc = [len(set(l) & r) / len(r) for l, r in zip(lists, rel_sets)]
    return {f"Precision@{k}": float(np.mean(p)), f"Recall@{k}": float(np.mean(rc)),
            f"HitRate@{k}": float(np.mean(h)), "users": len(lists)}


# ============================================================================ figures
def _save(fig, name):
    path = C.FIG_DIR / name
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_activation_study(curves, dataset, name):
    """curves = {activation: history}. Replicates paper Fig. 3 (a) training, (b) validation."""
    colours = dict(relu=BLUE, tanh=ORANGE, elu=AQUA, selu=YELLOW)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, key, title in zip(axes, ["train_mse", "val_mse"], ["(a) Training", "(b) Validation"]):
        for act, h in curves.items():
            y = h[key]
            ax.plot(np.arange(1, len(y) + 1), y, color=colours[act], lw=2, label=act.upper())
        ax.set_title(f"{title} masked MSE - {dataset}")
        ax.set_xlabel("epoch")
        ax.set_ylabel("masked MSE")
        lo = min(min(h[key]) for h in curves.values())
        ax.set_ylim(lo * 0.95, lo * 2.2)       # zoom on the converged region
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.06))
    return _save(fig, name)


def plot_learning_curve(h, dataset, name):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(np.arange(1, len(h["train_mse"]) + 1), h["train_mse"], color=BLUE, lw=2, label="train")
    if h["val_mse"]:
        ax.plot(np.arange(1, len(h["val_mse"]) + 1), h["val_mse"], color=ORANGE, lw=2,
                label="validation")
        b = h["best_epoch"]
        ax.axvline(b, color=GREY, lw=1, ls="--")
        ax.annotate(f"best epoch {b}", (b, max(h["val_mse"])), xytext=(4, 0),
                    textcoords="offset points", color=INK2, fontsize=8)
    ax.set_title(f"RikoNet autoencoder learning curve - {dataset}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("masked MSE")
    ax.legend()
    return _save(fig, name)


def plot_metric_bars(rows, metric, title, name, highlight="RikoNet", paper=None):
    """Horizontal bar chart, sorted, our model in blue, baselines grey, paper value orange."""
    rows = sorted(rows, key=lambda r: r[metric], reverse=True)
    labels = [r["model"] for r in rows]
    vals = [r[metric] for r in rows]
    cols = [BLUE if highlight in l else GREY for l in labels]
    if paper is not None:
        labels.append("RikoNet (paper, reported)")
        vals.append(paper)
        cols.append(ORANGE)
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(labels) + 1))
    y = np.arange(len(labels))
    ax.barh(y, vals, color=cols, height=0.62)
    for yi, v in zip(y, vals):
        ax.text(v, yi, f" {v:.3f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, labels)
    ax.set_xlabel(f"{metric} (lower is better)" if metric in ("MSE", "RMSE", "MAE") else metric)
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(vals) * 1.15)
    return _save(fig, name)


def plot_rating_distribution(data, name):
    fig, ax = plt.subplots(figsize=(6, 3.4))
    v = data.ratings.r.value_counts().sort_index()
    ax.bar(v.index, v.values / 1e3, color=BLUE, width=0.7)
    ax.set_xticks(v.index)
    ax.set_xlabel("rating")
    ax.set_ylabel("ratings (thousands)")
    ax.set_title(f"Rating distribution - {data.name}")
    ax.grid(axis="x", visible=False)
    return _save(fig, name)


def plot_cluster_sizes(labels, name):
    sizes = np.bincount(labels)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.hist(sizes, bins=40, color=BLUE, edgecolor="white", linewidth=0.8)
    ax.set_xlabel("anime per cluster")
    ax.set_ylabel("number of clusters")
    ax.set_title(f"Spectral clusters: size distribution (k = {len(sizes)})")
    ax.grid(axis="x", visible=False)
    return _save(fig, name)


def plot_embedding_map(xy, labels, highlight, names, name):
    """2-D t-SNE of the anime embeddings; a few large clusters are coloured, the rest grey."""
    fig, ax = plt.subplots(figsize=(7, 5.6))
    ax.scatter(xy[:, 0], xy[:, 1], s=3, color=GREY, alpha=0.35, linewidths=0)
    for c, col in zip(highlight, [BLUE, ORANGE, AQUA]):
        m = labels == c
        ax.scatter(xy[m, 0], xy[m, 1], s=10, color=col, edgecolors="white", linewidths=0.4,
                   label=f"cluster {c}: {names[c]}")
    ax.set_title("t-SNE of RikoNet anime embeddings (3 largest clusters highlighted)")
    ax.set_xticks([]), ax.set_yticks([])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), fontsize=8)
    return _save(fig, name)


def plot_ranking_bars(rows, metric, title, name):
    return plot_metric_bars(rows, metric, title, name)
