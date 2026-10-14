"""
main.py - runs the whole RikoNet replication and writes every number/figure to outputs/.

Experiments
  E1  MovieLens-100K, paper protocol (5 % of each user's ratings held out)  -> paper Table 4
  E2  MovieLens-100K, standard 80/20 protocol                               -> robustness check
  E3  Activation-function study (ReLU / tanh / ELU / SELU)                  -> paper Fig. 3
  E4  Anime (Kaggle), paper protocol: rating prediction + top-K metrics
  E5  Anime: embeddings -> self-tuning table -> spectral clustering -> opposite clusters
      -> hybrid filter ('Similar Anime' / 'Anime You May Like') evaluation  -> paper Sec. 3.1.2-3.1.5

Usage:  python main.py            (full run, roughly 2 h on a laptop CPU)
        python main.py --skip-anime  (MovieLens only)
        python main.py --anime-only  (anime only, merged into existing results.json)
        python main.py --quick    (tiny smoke test, a few minutes)
"""
import argparse
import json
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import pandas as pd

import baselines as B
import config as C
import data_prep as D
import evaluate as E
import recommend as R
from model import DeepAutoencoder

PAPER_ML100K = {"MSE": 0.349, "RMSE": 0.591}   # paper Table 4, "Proposed Model: RikoNet"
KG_DEFAULT = dict(hidden=[512, 512, 1024], dropout=0.8, refeed=True, l2=0.0)


def load_cfg(name):
    p = C.OUT_DIR / f"best_config_{name}.json"
    return json.loads(p.read_text()) if p.exists() else dict(KG_DEFAULT)


def new_ae(data, cfg, seed, activation=C.ACTIVATION):
    return DeepAutoencoder(data.n_items, n_side=0 if data.side is None else data.side.shape[1],
                           hidden=tuple(cfg["hidden"]), dropout=cfg["dropout"],
                           refeed=cfg["refeed"], l2=cfg["l2"], activation=activation, seed=seed)


# ============================================================================ rating experiments
def rating_experiment(data, cfgs, seeds, rel_threshold, tag, quick=False, epochs=C.MAX_EPOCHS,
                      patience=C.PATIENCE):
    """Train RikoNet autoencoder(s) and all baselines on the same split; return result tables."""
    Xtr, Xv, Xte = data.matrix("train"), data.matrix("val"), data.matrix("test")
    Xknown = data.matrix("train", "val")
    rating_rows, rank_rows, keep = [], [], {}

    for label, cfg in cfgs.items():
        runs, rruns = [], []
        for seed in seeds:
            t0 = time.time()
            ae = new_ae(data, cfg, seed)
            h = ae.fit(Xtr, Xv, data.side, label=f"{tag}/{label}/seed{seed}",
                       epochs=8 if quick else epochs, patience=patience)
            pr = lambda rows, ae=ae: ae.predict(Xtr, data.side, rows)
            m = E.rating_metrics(pr, Xte, data.r_min, data.r_max)
            m.update(seed=seed, best_epoch=h["best_epoch"], seconds=round(time.time() - t0))
            runs.append(m)
            rruns.append(E.ranking_metrics(pr, Xknown, Xte, rel_threshold))
            if seed == seeds[0]:
                keep[label] = (ae, h)
            print(f"  {label} seed {seed}: RMSE {m['RMSE']:.4f}  MAE {m['MAE']:.4f}", flush=True)
        row = {"model": label, "config": json.dumps(cfg)}
        for k in ("MSE", "RMSE", "MAE"):
            row[k] = float(np.mean([r[k] for r in runs]))
            row[k + "_std"] = float(np.std([r[k] for r in runs]))
        row["runs"] = len(runs)
        row["best_epoch"] = float(np.mean([r["best_epoch"] for r in runs]))
        rating_rows.append(row)
        rr = {"model": label}
        for k in rruns[0]:
            rr[k] = float(np.mean([r[k] for r in rruns]))
        rank_rows.append(rr)

    rows_all = np.arange(data.n_users)
    for bl in B.rating_baselines() + [B.Popularity(), B.RandomRanker()]:
        t0 = time.time()
        bl.fit(Xtr)
        if not isinstance(bl, (B.Popularity, B.RandomRanker)):
            m = E.rating_metrics(bl.predict_rows, Xte, data.r_min, data.r_max)
            rating_rows.append({"model": bl.name, **{k: m[k] for k in ("MSE", "RMSE", "MAE")},
                                "runs": 1})
            print(f"  {bl.name}: RMSE {m['RMSE']:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        rank_rows.append({"model": bl.name,
                          **E.ranking_metrics(bl.predict_rows, Xknown, Xte, rel_threshold)})
    return pd.DataFrame(rating_rows), pd.DataFrame(rank_rows), keep


def activation_study(data, cfg, epochs, tag):
    Xtr, Xv = data.matrix("train"), data.matrix("val")
    curves = {}
    for act in C.ACTIVATIONS_TO_COMPARE:
        ae = new_ae(data, cfg, C.SEED, activation=act)
        curves[act] = ae.fit(Xtr, Xv, data.side, epochs=epochs, patience=10 ** 6,
                             label=f"{tag}/{act}", verbose=False)
        print(f"  activation {act:5s}: best val MSE {min(curves[act]['val_mse']):.4f}", flush=True)
    return curves


# ============================================================================ anime pipeline
def genre_jaccard_pairs(G, a, b):
    inter = (G[a] * G[b]).sum(1)
    union = G[a].sum(1) + G[b].sum(1) - inter
    return inter / np.maximum(union, 1)


def cluster_genre_coherence(G, labels, rng, max_pairs=200_000):
    """Mean genre-Jaccard of random same-cluster pairs vs random pairs (higher = more coherent)."""
    n = len(labels)
    a, b = rng.integers(0, n, max_pairs), rng.integers(0, n, max_pairs)
    same = labels[a] == labels[b]
    same &= a != b
    return float(genre_jaccard_pairs(G, a[same], b[same]).mean()), \
        float(genre_jaccard_pairs(G, a, b).mean())


def anime_recommendation_pipeline(data, ae, results, quick=False):
    from sklearn.cluster import KMeans
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score

    rng = np.random.default_rng(C.SEED)
    Xtr, Xv, Xte = data.matrix("train"), data.matrix("val"), data.matrix("test")
    G = data.item_genres

    print("\n[E5] embeddings", flush=True)
    emb, info = R.build_embeddings(data, Xtr, Xv)
    np.save(C.MODEL_DIR / "anime_embeddings.npy", emb)

    print("[E5] self-tuning table (paper Table 1 analogue)", flush=True)
    table1 = R.self_tuning_table(emb, range(1, 4) if quick else [1, 2, 3, 4, 5, 7, 10, 15])
    table1.to_csv(C.OUT_DIR / "table1_self_tuning.csv")

    print("[E5] spectral clustering", flush=True)
    n_neighbors = C.MIN_CLUSTER_ELEMS
    t0 = time.time()
    labels = R.cluster_anime(emb, C.N_CLUSTERS, n_neighbors)
    t_spec = time.time() - t0
    km = KMeans(C.N_CLUSTERS, n_init=4, random_state=C.SEED).fit_predict(emb)
    sizes = np.bincount(labels, minlength=C.N_CLUSTERS)

    coh_s, coh_rand = cluster_genre_coherence(G, labels, rng)
    coh_k, _ = cluster_genre_coherence(G, km, rng)
    clus = {
        "n_clusters": C.N_CLUSTERS, "n_neighbors": n_neighbors, "seconds": round(t_spec),
        "top_users_m": info["top_users"], "embedding_dim": int(emb.shape[1]),
        "size_min": int(sizes.min()), "size_median": float(np.median(sizes)),
        "size_max": int(sizes.max()), "empty_clusters": int((sizes == 0).sum()),
        "clusters_below_4": int((sizes < 4).sum()),
        "silhouette_spectral": float(silhouette_score(emb, labels, sample_size=4000, random_state=0)),
        "silhouette_kmeans": float(silhouette_score(emb, km, sample_size=4000, random_state=0)),
        "genre_jaccard_within_spectral": coh_s, "genre_jaccard_within_kmeans": coh_k,
        "genre_jaccard_random_pairs": coh_rand,
    }

    print("[E5] logically opposite clusters (Algorithm 1)", flush=True)
    opp = R.opposite_titles(emb)
    opp_cluster = labels[opp]
    idx = np.arange(len(opp))
    clus["genre_jaccard_title_vs_opposite"] = float(genre_jaccard_pairs(G, idx, opp).mean())
    clus["genre_jaccard_title_vs_random"] = float(
        genre_jaccard_pairs(G, idx, rng.permutation(len(opp))).mean())
    clus["opposite_cluster_equals_own"] = float((opp_cluster == labels).mean())
    results["clustering"] = clus
    print(json.dumps(clus, indent=1))

    # cluster descriptions = 2 most frequent genres
    cl_names = {}
    for c in range(C.N_CLUSTERS):
        m = labels == c
        top = np.argsort(-G[m].sum(0))[:2] if m.any() else []
        cl_names[c] = " + ".join(data.genre_names[g] for g in top)

    # ---- knowledge base for recommend.py --------------------------------------------------
    np.savez(C.MODEL_DIR / "anime_kb.npz", labels=labels, opp_cluster=opp_cluster, opp=opp,
             names=np.array(data.item_names, dtype=object), item_ids=data.item_ids, genres=G,
             genre_names=np.array(data.genre_names, dtype=object), members=data.item_members)

    # ---- hybrid filter evaluation ----------------------------------------------------------
    print("[E5] hybrid filter evaluation", flush=True)
    df = data.ratings
    known = df[df.split != "test"].sort_values("order")
    by_user = {u: (g.i.values, g.r.values) for u, g in known.groupby("u")}
    test = df[(df.split == "test") & (df.r >= C.RELEVANT_ANIME)]
    rel = test.groupby("u").i.apply(set)
    users = rel.index.values
    if quick:
        users = users[:500]
    sims, mays, plain, rels, jac_s, jac_m, jac_p, csize = [], [], [], [], [], [], [], []
    for s in range(0, len(users), 1024):
        rows = users[s:s + 1024]
        P = np.clip(ae.predict(Xtr, None, rows), 1, 10)
        for j, u in enumerate(rows):
            items, vals = by_user[u]
            # temporal context: Kaggle has no timestamps, so we take 3 random known titles
            ctx = rng.choice(len(items), C.LAST_N_RATED, replace=False)
            order = np.r_[np.setdiff1d(np.arange(len(items)), ctx), ctx]
            sim, may, chosen = R.hybrid_filter(P[j], items[order], vals[order], labels, opp_cluster)
            p = P[j].copy()
            p[items] = -np.inf
            top = list(np.argsort(-p)[:C.TOP_K])
            sims.append(sim); mays.append(may); plain.append(top); rels.append(rel[u])
            csize.append(int(np.isin(labels, chosen).sum()))
            liked_ctx = items[ctx]
            for lst, acc in ((sim, jac_s), (may, jac_m), (top, jac_p)):
                if lst:
                    acc.append(float(np.mean([genre_jaccard_pairs(G, np.full(len(lst), c), np.array(lst)).mean()
                                              for c in liked_ctx])))
    hyb = {
        "Autoencoder top-10 (no filter)": E.list_precision(plain, rels),
        "Hybrid: Similar Anime": E.list_precision(sims, rels),
        "Hybrid: Anime You May Like": E.list_precision(mays, rels),
        "Hybrid: both lists (20 titles)": E.list_precision([a + b for a, b in zip(sims, mays)],
                                                           rels, k=2 * C.TOP_K),
    }
    for k, acc in (("Autoencoder top-10 (no filter)", jac_p), ("Hybrid: Similar Anime", jac_s),
                   ("Hybrid: Anime You May Like", jac_m)):
        hyb[k]["genre_jaccard_to_context"] = float(np.mean(acc))
    hyb["Hybrid: Similar Anime"]["mean_list_length"] = float(np.mean([len(x) for x in sims]))
    hyb["Hybrid: Similar Anime"]["mean_candidate_set_C"] = float(np.mean(csize))
    results["hybrid_filter"] = hyb
    print(json.dumps(hyb, indent=1))

    # ---- example for one real test user -------------------------------------------------------
    u = users[0]
    items, vals = by_user[u]
    P = np.clip(ae.predict(Xtr, None, [u])[0], 1, 10)
    sim, may, chosen = R.hybrid_filter(P, items, vals, labels, opp_cluster)
    ex = pd.concat([
        pd.DataFrame({"list": "context (last 3 known ratings)", "anime": [data.item_names[i] for i in items[-3:]],
                      "rating / predicted": vals[-3:], "cluster": labels[items[-3:]]}),
        pd.DataFrame({"list": "Similar Anime", "anime": [data.item_names[i] for i in sim],
                      "rating / predicted": P[sim].round(2), "cluster": labels[sim]}),
        pd.DataFrame({"list": "Anime You May Like", "anime": [data.item_names[i] for i in may],
                      "rating / predicted": P[may].round(2), "cluster": labels[may]})])
    ex["anime"] = ex.anime.map(__import__("html").unescape)
    ex.to_csv(C.OUT_DIR / "example_recommendations.csv", index=False)
    results["example_user_kaggle_id"] = int(data.user_ids[u])

    # ---- figures ------------------------------------------------------------------------------
    E.plot_cluster_sizes(labels, "anime_cluster_sizes.png")
    xy = TSNE(2, random_state=C.SEED, init="pca", perplexity=30).fit_transform(emb)
    # highlight the 3 largest clusters, skipping adult-genre clusters (classroom-friendly figure)
    big = [c for c in np.argsort(-sizes) if "Hentai" not in cl_names[c]][:3]
    E.plot_embedding_map(xy, labels, big, cl_names, "anime_embedding_tsne.png")
    pd.DataFrame({"cluster": range(C.N_CLUSTERS), "size": sizes,
                  "top_genres": [cl_names[c] for c in range(C.N_CLUSTERS)]}) \
        .to_csv(C.OUT_DIR / "anime_clusters.csv", index=False)


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-anime", action="store_true")
    ap.add_argument("--anime-activation", action="store_true", help="also run E3 on anime (slow)")
    ap.add_argument("--anime-only", action="store_true",
                    help="run E3b/E4/E5 only and merge into an existing outputs/results.json")
    ap.add_argument("--seeds", type=int, default=3, help="repeats for MovieLens (mean +- std)")
    args = ap.parse_args()
    t_start = time.time()
    results = {"config": {k: (list(v) if isinstance(v, (tuple, range)) else v)
                          for k, v in vars(C).items() if k.isupper() and not k.endswith("DIR")}}
    seeds = [C.SEED + s for s in range(1 if args.quick else args.seeds)]
    ml_cfg = load_cfg("movielens")
    results["config"]["movielens_tuned"] = ml_cfg
    if args.anime_only:
        prev = json.loads((C.OUT_DIR / "results.json").read_text())
        prev.update({k: v for k, v in results.items() if k != "config"})
        prev["config"].update(results["config"])
        results = prev
        t_start -= results.get("runtime_minutes", 0) * 60
    else:
        run_movielens(results, ml_cfg, seeds, args, t_start)
    if args.skip_anime:
        return
    run_anime(results, seeds, args, t_start)


def run_movielens(results, ml_cfg, seeds, args, t_start):

    # ---------------- E1 MovieLens paper protocol
    print("\n[E1] MovieLens-100K, paper protocol (5 % held-out per user)", flush=True)
    ml = D.load_movielens()
    results["movielens_stats"] = ml.stats
    E.plot_rating_distribution(ml, "movielens_rating_distribution.png")
    cfgs = {"RikoNet AE (tuned on validation)": ml_cfg,
            "RikoNet AE (K&G default config)": KG_DEFAULT}
    rt, rk, keep = rating_experiment(ml, cfgs, seeds, C.RELEVANT_MOVIELENS, "E1", args.quick)
    rt.to_csv(C.OUT_DIR / "table_E1_movielens_paper_protocol.csv", index=False)
    rk.to_csv(C.OUT_DIR / "table_E1_movielens_topk.csv", index=False)
    results["E1"] = {"rating": rt.to_dict("records"), "ranking": rk.to_dict("records"),
                     "paper_reported": PAPER_ML100K}
    E.plot_learning_curve(keep["RikoNet AE (tuned on validation)"][1], "MovieLens-100K",
                          "movielens_learning_curve.png")
    E.plot_metric_bars(rt.to_dict("records"), "RMSE",
                       "MovieLens-100K, paper protocol: test RMSE", "movielens_rmse_comparison.png",
                       paper=PAPER_ML100K["RMSE"])
    E.plot_metric_bars(rk.to_dict("records"), "NDCG@10",
                       "MovieLens-100K: NDCG@10 (higher is better)", "movielens_ndcg.png")

    # ---------------- E2 MovieLens 80/20
    print("\n[E2] MovieLens-100K, 80/20 protocol", flush=True)
    ml2 = D.load_movielens(test_frac=C.STANDARD_TEST_FRACTION)
    rt2, rk2, _ = rating_experiment(ml2, {"RikoNet AE (tuned on validation)": ml_cfg}, seeds,
                                    C.RELEVANT_MOVIELENS, "E2", args.quick)
    rt2.to_csv(C.OUT_DIR / "table_E2_movielens_80_20.csv", index=False)
    rk2.to_csv(C.OUT_DIR / "table_E2_movielens_80_20_topk.csv", index=False)
    results["E2"] = {"rating": rt2.to_dict("records"), "ranking": rk2.to_dict("records")}
    E.plot_metric_bars(rt2.to_dict("records"), "RMSE", "MovieLens-100K, 80/20 split: test RMSE",
                       "movielens_8020_rmse_comparison.png")

    # ---------------- E3 activation study (MovieLens)
    print("\n[E3] activation study, MovieLens", flush=True)
    curves = activation_study(ml, ml_cfg, 8 if args.quick else C.ACTIVATION_STUDY_EPOCHS, "E3")
    results["E3_movielens"] = {a: {"best_val_mse": min(h["val_mse"]),
                                   "best_epoch": int(np.argmin(h["val_mse"]) + 1)}
                               for a, h in curves.items()}
    E.plot_activation_study(curves, "MovieLens-100K", "activation_study_movielens.png")
    _dump(results, t_start)


def run_anime(results, seeds, args, t_start):
    # ---------------- E4 anime rating prediction
    print("\n[E4] Anime (Kaggle), paper protocol", flush=True)
    an = D.load_anime() if not args.quick else D.load_anime(n_users=2000, n_items=1500)
    results["anime_stats"] = an.stats
    E.plot_rating_distribution(an, "anime_rating_distribution.png")
    an_cfg = load_cfg("anime")
    results["config"]["anime_tuned"] = an_cfg
    rt4, rk4, keep4 = rating_experiment(an, {"RikoNet AE (tuned on validation)": an_cfg},
                                        seeds[:1], C.RELEVANT_ANIME, "E4", args.quick,
                                        epochs=60, patience=6)   # time budget on a laptop CPU
    rt4.to_csv(C.OUT_DIR / "table_E4_anime_rating.csv", index=False)
    rk4.to_csv(C.OUT_DIR / "table_E4_anime_topk.csv", index=False)
    results["E4"] = {"rating": rt4.to_dict("records"), "ranking": rk4.to_dict("records")}
    ae, h = keep4["RikoNet AE (tuned on validation)"]
    E.plot_learning_curve(h, "Anime (Kaggle)", "anime_learning_curve.png")
    E.plot_metric_bars(rt4.to_dict("records"), "RMSE", "Anime (Kaggle), paper protocol: test RMSE",
                       "anime_rmse_comparison.png")
    E.plot_metric_bars(rk4.to_dict("records"), "NDCG@10", "Anime (Kaggle): NDCG@10 (higher is better)",
                       "anime_ndcg.png")
    ae.model.save_weights(C.MODEL_DIR / "anime_primary_ae.weights.h5")
    (C.MODEL_DIR / "anime_ae_config.json").write_text(json.dumps(an_cfg))
    _dump(results, t_start)

    # ---------------- E3b activation study on anime (optional: ~30 extra minutes on CPU)
    if args.anime_activation:
        print("\n[E3b] activation study, anime", flush=True)
        curves = activation_study(an, an_cfg, 6 if args.quick else 25, "E3b")
        results["E3_anime"] = {a: {"best_val_mse": min(h["val_mse"]),
                                   "best_epoch": int(np.argmin(h["val_mse"]) + 1)}
                               for a, h in curves.items()}
        E.plot_activation_study(curves, "Anime (Kaggle)", "activation_study_anime.png")
        _dump(results, t_start)

    # ---------------- E5 embeddings, clusters, hybrid filter
    anime_recommendation_pipeline(an, ae, results, args.quick)
    _dump(results, t_start)


def _dump(results, t_start):
    results["runtime_minutes"] = round((time.time() - t_start) / 60, 1)
    (C.OUT_DIR / "results.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"-> outputs/results.json updated ({results['runtime_minutes']} min)", flush=True)


if __name__ == "__main__":
    main()
