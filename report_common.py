"""
report_common.py - loads results.json and turns it into the facts used by BOTH the Word report
and the PowerPoint deck, so the two documents can never disagree with each other or with the
experiment outputs. Nothing in here is typed by hand except the paper's own published numbers.
"""
import json

import pandas as pd

import config as C

PAPER = dict(
    title="RikoNet: A Novel Anime Recommendation Engine",
    authors="Badal Soni, Debangan Thakuria, Nilutpal Nath, Navarun Das, Bhaskarananda Boro",
    venue="Multimedia Tools and Applications, vol. 82, pp. 32329-32348 (2023), Springer",
    doi="10.1007/s11042-023-14710-9",
    url="https://link.springer.com/article/10.1007/s11042-023-14710-9",
    repo="https://github.com/NilutpalNath/RikoNet",
)

# Paper Table 4 (MovieLens-100K), copied verbatim from the published article
PAPER_TABLE4 = [
    ("Global Average", 1.129, 1.062), ("User Average", 0.929, 0.964),
    ("User KNN - Cosine", 0.813, 0.813), ("Item KNN - Cosine", 0.854, 0.862),
    ("User KNN - Pearson", 0.814, 0.902), ("Item KNN - Pearson", 0.796, 0.892),
    ("SVD", 0.879, 0.937), ("Biased MF", 0.768, 0.877), ("SVD++", 0.756, 0.869),
    ("DNNRec", 0.747, 0.864), ("UBP", 0.724, 0.851), ("PDMF-post 50D", 0.427, 0.654),
    ("Proposed Model: RikoNet", 0.349, 0.591),
]

OURS = "RikoNet AE (tuned on validation)"
KG = "RikoNet AE (K&G default config)"


def load():
    R = json.loads((C.OUT_DIR / "results.json").read_text())
    F = {}
    for e in ("E1", "E2", "E4"):
        if e in R:
            F[e] = pd.DataFrame(R[e]["rating"]).sort_values("RMSE").reset_index(drop=True)
            F[e + "_rank"] = pd.DataFrame(R[e]["ranking"])
    return R, F


def row(df, model):
    return df[df.model == model].iloc[0]


def best_other(df, exclude="RikoNet"):
    d = df[~df.model.str.contains(exclude)]
    return d.sort_values("RMSE").iloc[0]


def compare(a, b, lower_better=True):
    """'better' / 'worse' / 'equal' for a vs b."""
    if abs(a - b) < 5e-4:
        return "about equal to"
    return ("lower (better) than" if a < b else "higher (worse) than") if lower_better else \
        ("higher (better) than" if a > b else "lower (worse) than")


def fmt(x, d=3):
    return f"{x:.{d}f}"


def facts():
    """Short computed statements used in both documents."""
    R, F = load()
    out = {"R": R, "F": F}
    e1 = F["E1"]
    ours = row(e1, OURS)
    bo = best_other(e1)
    out["e1_ours"] = ours
    out["e1_best_baseline"] = bo
    out["e1_vs_baseline"] = compare(ours.RMSE, bo.RMSE)
    out["e1_gap_to_paper"] = ours.RMSE - 0.591
    if "E2" in F:
        out["e2_ours"] = row(F["E2"], OURS)
        out["e2_best_baseline"] = best_other(F["E2"])
    for key in ("E3_movielens", "E3_anime"):
        if key in R:
            best = min(R[key], key=lambda a: R[key][a]["best_val_mse"])
            out[key + "_best"] = best
            out[key + "_selu_rank"] = sorted(R[key], key=lambda a: R[key][a]["best_val_mse"]).index("selu") + 1
    if "E4" in F:
        out["e4_ours"] = row(F["E4"], OURS)
        out["e4_best_baseline"] = best_other(F["E4"])
        rk = F["E4_rank"].sort_values("NDCG@10", ascending=False)
        out["e4_best_ranker"] = rk.iloc[0]
        out["e4_ours_rank"] = row(F["E4_rank"], OURS)
    return out


SHORT = {OURS: "RikoNet AE (tuned)", KG: "RikoNet AE (K&G default)",
         "Bias baseline (mu+b_u+b_i)": "Bias baseline", "Biased MF (ALS, f=32)": "Biased MF (ALS)",
         "Item KNN (cosine, k=40)": "Item KNN", "User KNN (cosine, k=40)": "User KNN"}


def short(name):
    return SHORT.get(name, name)


def seeds_text(n):
    n = int(n)
    return f"{n} seed" if n == 1 else f"mean of {n} seeds"
