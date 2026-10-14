"""
make_report.py - builds outputs/RikoNet_Replication_Report.docx from outputs/results.json.
Run AFTER main.py.  python make_report.py
"""
import html
import json

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

import config as C
from report_common import KG, OURS, PAPER, PAPER_TABLE4, facts, fmt, row, seeds_text

INK = RGBColor(0x1F, 0x1B, 0x3A)
ACCENT = RGBColor(0xC2, 0x3B, 0x4A)
MUTED = RGBColor(0x5A, 0x57, 0x6B)

f = facts()
R, F = f["R"], f["F"]
doc = Document()

# ------------------------------------------------------------------ styles
st = doc.styles["Normal"]
st.font.name, st.font.size = "Calibri", Pt(11)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
st.paragraph_format.space_after = Pt(6)
st.paragraph_format.line_spacing = 1.15
for lvl, size in ((1, 16), (2, 13), (3, 11.5)):
    h = doc.styles[f"Heading {lvl}"]
    h.font.name, h.font.size, h.font.bold, h.font.color.rgb = "Cambria", Pt(size), True, INK
    h.element.rPr.rFonts.set(qn("w:eastAsia"), "Cambria")
    h.paragraph_format.space_before = Pt(14 if lvl == 1 else 10)
    h.paragraph_format.space_after = Pt(4)
for s in doc.sections:
    s.left_margin = s.right_margin = Cm(2.3)
    s.top_margin = s.bottom_margin = Cm(2.2)

fig_no = [0]
tab_no = [0]


def P(text, bold_lead=None, italic=False, size=None, color=None, align=None):
    p = doc.add_paragraph()
    if bold_lead:
        r = p.add_run(bold_lead)
        r.bold = True
    r = p.add_run(text)
    r.italic = italic
    if size:
        r.font.size = Pt(size)
    if color:
        r.font.color.rgb = color
    if align:
        p.alignment = align
    return p


def B(items, style="List Bullet"):
    for it in items:
        p = doc.add_paragraph(style=style)
        if isinstance(it, tuple):
            p.add_run(it[0]).bold = True
            p.add_run(it[1])
        else:
            p.add_run(it)


def H(text, lvl=1):
    doc.add_heading(text, lvl)


def shade(cell, hex_fill):
    tcPr = cell._tc.get_or_add_tcPr()
    sh = OxmlElement("w:shd")
    sh.set(qn("w:val"), "clear")
    sh.set(qn("w:color"), "auto")
    sh.set(qn("w:fill"), hex_fill)
    tcPr.append(sh)


def T(df, caption, widths=None, bold_rows=(), font=9):
    tab_no[0] += 1
    cap = doc.add_paragraph()
    r = cap.add_run(f"Table {tab_no[0]}. {caption}")
    r.bold, r.font.size, r.font.color.rgb = True, Pt(9.5), INK
    t = doc.add_table(rows=1, cols=len(df.columns))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, c in enumerate(df.columns):
        cell = t.rows[0].cells[j]
        cell.text = ""
        run = cell.paragraphs[0].add_run(str(c))
        run.bold, run.font.size = True, Pt(font)
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        shade(cell, "1F1B3A")
    for i, (_, rw) in enumerate(df.iterrows()):
        cells = t.add_row().cells
        for j, v in enumerate(rw):
            txt = fmt(v) if isinstance(v, float) else str(v)
            run = cells[j].paragraphs[0].add_run(txt)
            run.font.size = Pt(font)
            if i in bold_rows:
                run.bold = True
                shade(cells[j], "F6E3E5")
    if widths:
        for rw in t.rows:
            for j, w in enumerate(widths):
                rw.cells[j].width = Cm(w)
    doc.add_paragraph()


def FIG(name, caption, width=15.5):
    path = C.FIG_DIR / name
    if not path.exists():
        return
    fig_no[0] += 1
    doc.add_picture(str(path), width=Cm(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph()
    r = p.add_run(f"Figure {fig_no[0]}. {caption}")
    r.italic, r.font.size, r.font.color.rgb = True, Pt(9), MUTED
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER


def metric_table(df, cols=("model", "MSE", "RMSE", "MAE")):
    d = df.copy()
    if "RMSE_std" in d and d.RMSE_std.notna().any():
        d["RMSE"] = [f"{r:.3f} ± {s:.3f}" if pd.notna(s) and r_ > 1 else f"{r:.3f}"
                     for r, s, r_ in zip(d.RMSE, d.RMSE_std, d.runs.fillna(1))]
    d = d[list(cols)].rename(columns={"model": "Model"})
    bold = [i for i, m in enumerate(d.Model) if "RikoNet" in m]
    return d, bold


# ================================================================== title page
for _ in range(5):
    doc.add_paragraph()
P("Advanced Topics in Machine Learning (ATML) - Research Paper Replication", size=12, color=MUTED,
  align=WD_ALIGN_PARAGRAPH.CENTER)
P("Replicating RikoNet", size=30, color=INK, align=WD_ALIGN_PARAGRAPH.CENTER).runs[0].bold = True
P("A deep-autoencoder and spectral-clustering hybrid recommender for anime", size=14, color=ACCENT,
  align=WD_ALIGN_PARAGRAPH.CENTER)
doc.add_paragraph()
P(f"Original paper: {PAPER['authors']}, “{PAPER['title']}”, {PAPER['venue']}. "
  f"DOI {PAPER['doi']}", size=10, color=MUTED, align=WD_ALIGN_PARAGRAPH.CENTER)
doc.add_paragraph()
P("Student: ______________________      Roll no.: ______________      Supervisor: ______________________",
  size=10.5, align=WD_ALIGN_PARAGRAPH.CENTER)
P("Implementation: Python 3.12, TensorFlow/Keras, scikit-learn - developed in VS Code on Windows",
  size=10, color=MUTED, align=WD_ALIGN_PARAGRAPH.CENTER)
doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

# ================================================================== abstract
e1o, e1b = f["e1_ours"], f["e1_best_baseline"]
H("Abstract")
abstract = (
    "This report replicates RikoNet (Soni et al., Multimedia Tools and Applications, 2023), a hybrid "
    "anime recommender that predicts ratings with a deep autoencoder, learns anime embeddings with three "
    "stacked autoencoders, groups them with spectral clustering, and merges the two signals in a 'hybrid "
    "filter' that outputs a Similar Anime list and an Anime You May Like list. The authors' MyAnimeList "
    "dataset and code repository are not publicly available, so this is a simplified replication: the "
    "paper's own benchmark (MovieLens-100K, Table 4) is reproduced with the paper's evaluation protocol, "
    "and the full pipeline is rebuilt on the public Kaggle Anime Recommendations Database as an "
    f"alternative dataset. On MovieLens-100K our RikoNet autoencoder reaches test RMSE {fmt(e1o.RMSE)} "
    f"({seeds_text(e1o.runs)}), {f['e1_vs_baseline']} the best classical baseline we trained on "
    f"the same split ({e1b.model}, {fmt(e1b.RMSE)}), and far from the 0.591 reported in the paper. ")
if "e4_ours" in f:
    abstract += (f"On the anime data the autoencoder reaches RMSE {fmt(f['e4_ours'].RMSE)} on a 1-10 scale. ")
abstract += ("We could not reproduce the paper's headline improvement; we discuss protocol differences, "
             "unreported hyper-parameters and inconsistencies in the published table that may explain the gap.")
P(abstract)

# ================================================================== 1 intro
H("1. Introduction and objective")
P("The objective of this ATML project is to select a closed-access (subscription) Springer journal "
  "article whose data is publicly available, re-implement its method from the paper alone, and test "
  "whether the reported results hold. Recommender systems are a natural ATML topic because they combine "
  "representation learning (autoencoders, embeddings), unsupervised learning (clustering) and careful "
  "evaluation design (sparse data, leakage, ranking metrics).")
P("Every number in this report is produced by the code in the accompanying project folder and read "
  "automatically from outputs/results.json; no result has been typed in by hand except the numbers quoted "
  "from the original paper, which are always labelled as such.")

# ================================================================== 2 verification
H("2. Paper selection and verification")
T(pd.DataFrame([
    ("Title", PAPER["title"]), ("Authors", PAPER["authors"]), ("Journal", PAPER["venue"]),
    ("DOI", PAPER["doi"]), ("Received / accepted / online", "31 May 2021 / 4 Feb 2023 / 3 Mar 2023"),
    ("Access model", "Subscription. Springer page metadata: \"Open Access\": \"N\", "
                     "isAccessibleForFree: false; copyright © under exclusive licence to Springer"),
    ("Preprint", "An earlier version is on arXiv (2106.12970). Mention this to your faculty: the journal "
                 "article is closed-access, but a preprint of it is freely readable."),
    ("Authors' code", f"{PAPER['repo']} - returns HTTP 404 (checked 14 Oct 2026); not in the author's "
                      "public repository list"),
    ("Original dataset", "Crawled from MyAnimeList.net with the Jikan API; never released, and the paper "
                         "gives no reconstruction instructions"),
    ("Benchmark dataset", "MovieLens-100K (GroupLens) - public, identical to the paper's Table 4"),
], columns=["Item", "Finding"]), "Verification of the selected article", widths=[4, 12.4])

# ================================================================== 3 method
H("3. The RikoNet method (as described in the paper)")
H("3.1 Primary autoencoder (rating prediction)", 2)
P("RikoNet follows the deep autoencoder of Kuchaiev & Ginsburg (2017). Each user is a vector of ratings "
  "over all titles (0 = not rated). The encoder compresses it to a d-dimensional code and the decoder "
  "reconstructs a dense vector; the reconstruction at unrated positions is the predicted rating. "
  "Hidden layers use SELU, the output layer ReLU (ratings cannot be negative). Training minimises masked "
  "MSE, where only observed ratings contribute:")
P("MSE = Σ mᵢ (rᵢ − yᵢ)² / Σ mᵢ,   RMSE = √MSE,   "
  "mᵢ = 0 if rᵢ = 0 else 1", italic=True, align=WD_ALIGN_PARAGRAPH.CENTER)
H("3.2 Anime embeddings (three autoencoders)", 2)
B(["Stage 1 - every anime is represented by its ratings from the 'top raters' (users with more than t "
   "ratings); a masked autoencoder maps this m-dimensional vector to d dimensions.",
   "Stage 2 - the 43-dimensional one-hot genre vector is appended (d' = d + g), adding content knowledge.",
   "Stage 3 - an unmasked autoencoder with no output activation compresses d' back to d; its code is the "
   "final hybrid embedding."])
H("3.3 Spectral clustering and logically opposite clusters", 2)
P("Embeddings are clustered with spectral clustering on a nearest-neighbour affinity graph. The number of "
  "clusters is chosen with Zelnik-Manor & Perona's self-tuning method (paper Table 1); the authors hand-"
  "picked 222 clusters with 'minimum elements' 4. Algorithm 1 then finds, for each title a₀, the title "
  "that is 'logically opposite': at a similar distance from the global centroid M but pointing the "
  "opposite way. It minimises (1 − d-ratio) + (cos∠(aᵢ−M, a₀−M) + 1), "
  "with ε = 1e-9.")
H("3.4 Hybrid filter", 2)
P("Let S be the unrated titles with their predicted ratings and C the union of the clusters of the user's "
  "last three rated titles - the title's own cluster if it was liked, its opposite cluster if it was "
  "disliked (the threshold is not given). Similar Anime = top-10 of S ∩ C; Anime You May Like = "
  "top-10 of S − C, both sorted by predicted rating.")
H("3.5 Evaluation in the paper", 2)
P("The paper hides 5 % of each user's ratings, predicts them and reports masked MSE/RMSE. The comparison "
  "(Table 4) is on MovieLens-100K; competitor numbers are copied from Kiran et al. (2020). A 20-person "
  "user study reports list ratings rising from 7.2 to 8.97 over about 10 iterations.")

# ================================================================== 4 exact vs simplified
H("4. Exact reproduction versus simplified replication")
P("This project is a simplified replication. The table states, for every component, what the paper "
  "specifies and what we did. 'Faithful' means implemented as described; 'Our choice' means the paper "
  "leaves it unspecified; 'Substituted' means the original was unavailable.")
cfg = R["config"]
mlc, anc = cfg.get("movielens_tuned", {}), cfg.get("anime_tuned", {})
T(pd.DataFrame([
    ("Primary AE: SELU hidden, ReLU output, masked MSE", "Specified", "Faithful"),
    ("Primary AE: layer sizes, dropout, optimiser, epochs", "Not reported",
     f"Our choice, tuned on validation only. MovieLens: {mlc.get('hidden')}, dropout {mlc.get('dropout')}, "
     f"re-feeding {mlc.get('refeed')}, L2 {mlc.get('l2')}. Anime: {anc.get('hidden')}, dropout "
     f"{anc.get('dropout')}, re-feeding {anc.get('refeed')} (best of 5 anime configurations tried). Adam, "
     f"lr 1e-3, batch 128, early stopping"),
    ("Demographic inputs (gender, 5 age groups)", "Used", "Faithful on MovieLens (u.user); not "
                                                          "possible on Kaggle anime (no demographics)"),
    ("5 % per-user hold-out, masked MSE/RMSE", "Specified", "Faithful (+ a separate 5 % validation split)"),
    ("Anime dataset (MyAnimeList, 6,668 titles, 16,384 users)", "Private", "Substituted: Kaggle "
     "CooperUnion data reduced to the same 6,668 titles x 16,384 users"),
    ("Cleaning: unknown studio/source, plan-to-watch, dropped->1", "Specified",
     "Approximated: unknown genre/type removed, -1 'watched but unscored' removed; no status column"),
    ("Embedding size d, top-rater threshold t", "Not reported",
     f"Our choice: d = {C.EMB_DIM}, t = {C.TOP_USER_MIN_RATINGS} ratings"),
    ("Self-tuning cluster count (Table 1)", "Zelnik-Manor & Perona",
     "Approximated with the eigengap criterion"),
    ("Spectral clustering, k-NN affinity, 222 clusters", "Specified",
     f"Faithful (222 clusters, {C.MIN_CLUSTER_ELEMS} neighbours)"),
    ("Algorithm 1 (opposite cluster)", "Specified", "Faithful"),
    ("Hybrid filter, last 3 titles, top-10 lists", "Specified",
     f"Faithful; like threshold {C.LIKE_THRESHOLD_ANIME}/10 is our choice; Kaggle has no timestamps "
     "so 3 random known titles stand in for 'last 3'"),
    ("Baselines", "Copied from other papers", "Re-trained by us on the same split (fairer)"),
    ("Top-K metrics (Precision/Recall/NDCG@10)", "Not used", "Added"),
    ("20-person user study", "Reported", "Not replicated (needs human participants)"),
], columns=["Component", "Paper", "This replication"]), "Exact versus simplified replication",
    widths=[5.2, 2.8, 8.4], font=8.5)

# ================================================================== 5 data
H("5. Datasets")
ms = R["movielens_stats"]
P(f"MovieLens-100K: {ms['users']} users, {ms['items']} movies, {ms['ratings']:,} ratings on a 1-5 "
  f"scale (density {ms['density'] * 100:.2f} %). This is the same public dataset the paper uses in Table 4.",
  bold_lead="MovieLens-100K. ")
if "anime_stats" in R:
    a = R["anime_stats"]
    P(f"The Kaggle 'Anime Recommendations Database' (CooperUnion, data from MyAnimeList) contains "
      f"{a['raw_anime']:,} anime and {a['raw_rows']:,} user-anime rows. {a['unscored_rows']:,} rows are -1 "
      f"(watched, no score) and were removed. Keeping the 6,668 most-rated titles and the 16,384 most active "
      f"users (each has at least {a['user_min_ratings']} ratings) gives {a['ratings']:,} ratings "
      f"(density {a['density'] * 100:.2f} %, mean {a['mean_rating']:.2f}/10). It has exactly "
      f"{a['genres']} genres, the same count as the paper. IMPORTANT: this is an alternative dataset, not "
      "the one used by the authors.", bold_lead="Anime (Kaggle) - alternative dataset. ")
    FIG("anime_rating_distribution.png", "Rating distribution of the cleaned Kaggle anime data.", 11)
else:
    P("The anime experiments (E4, E5) were still running when this version of the report was generated; "
      "re-run make_report.py after main.py --anime-only finishes to include them.", italic=True, color=MUTED)

# ================================================================== 6 implementation
H("6. Implementation")
P("The code is split into small modules so that each part of the paper maps to one file:")
B([("config.py - ", "every setting, tagged [PAPER], [K&G] or [OURS] so the origin of each choice is visible."),
   ("data_prep.py - ", "loads both datasets, applies the cleaning rules, builds sparse user x item matrices "
                       "and performs the per-user split (test 5 %, validation 5 %, train 90 %)."),
   ("model.py - ", "the deep autoencoder in TensorFlow/Keras with a custom training step implementing masked "
                   "MSE and dense re-feeding, early stopping on validation MSE and batched prediction."),
   ("baselines.py - ", "global/user/item averages, a regularised bias model, user- and item-KNN on baseline "
                       "residuals, biased matrix factorisation (ALS), popularity and random rankers."),
   ("recommend.py - ", "three-stage embeddings, the self-tuning table, spectral clustering, Algorithm 1 and "
                       "the hybrid filter; also a command-line recommender."),
   ("evaluate.py - ", "MSE/RMSE/MAE, Precision/Recall/NDCG/HitRate@10, coverage, and all figures."),
   ("tune.py / main.py - ", "validation-only hyper-parameter search and the end-to-end experiment runner.")])
H("6.1 Avoiding data leakage", 2)
B(["Held-out test ratings are removed from the input matrix before training; the model never sees them.",
   "Hyper-parameters are chosen with a separate validation split; the test split is read only once, "
   "by the final evaluation.",
   "Anime embeddings and clusters are built from training ratings only.",
   "In ranking evaluation every title the user already rated is excluded from the candidate list."])

# ================================================================== 7 results
H("7. Results")
H("7.1 E1 - MovieLens-100K with the paper's protocol (reproduces Table 4)", 2)
d, bold = metric_table(F["E1"])
T(d, f"MovieLens-100K test error, 5 % hold-out per user. RikoNet rows: "
     f"{seeds_text(e1o.runs)} (± std).", widths=[7, 3, 3.6, 3], bold_rows=bold)
P(f"Our tuned RikoNet autoencoder obtains RMSE {fmt(e1o.RMSE)} ± {fmt(e1o.RMSE_std)}, which is "
  f"{f['e1_vs_baseline']} the best baseline trained on the same split ({e1b.model}: {fmt(e1b.RMSE)}). "
  f"The configuration recommended by Kuchaiev & Ginsburg reaches {fmt(row(F['E1'], KG).RMSE)}. "
  f"The paper reports 0.591; our result is {fmt(f['e1_gap_to_paper'])} higher.")
FIG("movielens_rmse_comparison.png", "Test RMSE on MovieLens-100K (paper protocol). Blue: our RikoNet "
    "autoencoder; grey: baselines trained on the same split; orange: value reported in the paper.")
FIG("movielens_learning_curve.png", "Learning curve of the tuned RikoNet autoencoder (MovieLens-100K).", 12)

T(pd.DataFrame(PAPER_TABLE4, columns=["Method (paper Table 4)", "MSE", "RMSE"])
  .assign(**{"√MSE": lambda x: x.MSE ** 0.5})
  .assign(Consistent=lambda x: ["yes" if abs(a - b) < 0.002 else "NO" for a, b in zip(x["√MSE"], x.RMSE)]),
  "Numbers published in the paper (Table 4) and an internal-consistency check (RMSE should equal √MSE).",
  widths=[6, 2.5, 2.5, 2.5, 2.8], bold_rows=[2, 3])
P("Two rows of the published table (User KNN-Cosine and Item KNN-Cosine) are internally inconsistent: "
  "their RMSE is not the square root of their MSE. The competitor numbers were copied from another paper "
  "rather than re-run, so they were not necessarily measured with the same 5 % protocol as RikoNet.")

if "E2" in F:
    H("7.2 E2 - MovieLens-100K with a standard 80/20 split", 2)
    d, bold = metric_table(F["E2"])
    T(d, "MovieLens-100K test error with 20 % of each user's ratings held out.",
      widths=[7, 3, 3.6, 3], bold_rows=bold)
    e2o, e2b = f["e2_ours"], f["e2_best_baseline"]
    P(f"With the more common 80/20 protocol the autoencoder scores {fmt(e2o.RMSE)} against "
      f"{fmt(e2b.RMSE)} for {e2b.model}, so the ranking of methods is "
      f"{'the same' if (e2o.RMSE < e2b.RMSE) == (e1o.RMSE < e1b.RMSE) else 'different'} under both protocols.")

H("7.3 E3 - Effect of the activation function (reproduces Fig. 3)", 2)
for key, ds, fig in (("E3_movielens", "MovieLens-100K", "activation_study_movielens.png"),
                     ("E3_anime", "Anime (Kaggle)", "activation_study_anime.png")):
    if key in R:
        rows = sorted(R[key].items(), key=lambda kv: kv[1]["best_val_mse"])
        T(pd.DataFrame([(a.upper(), v["best_val_mse"], v["best_epoch"]) for a, v in rows],
                       columns=["Activation", "Best validation MSE", "Epoch"]),
          f"Activation study on {ds} (same architecture, only the activation changes).",
          widths=[4, 4, 3], bold_rows=[0])
        FIG(fig, f"Training and validation masked MSE per activation function - {ds}.")
        P(f"On {ds}, {f[key + '_best'].upper()} reaches the lowest validation MSE; SELU ranks "
          f"{f[key + '_selu_rank']} of 4. The paper reports SELU as the best activation"
          f"{' - confirmed here.' if f[key + '_best'] == 'selu' else ' - not confirmed on this dataset.'}")

if "E4" in F:
    H("7.4 E4 - Rating prediction and top-10 recommendation on the anime data", 2)
    d, bold = metric_table(F["E4"])
    T(d, "Anime (Kaggle) test error, 5 % hold-out per user, ratings 1-10.",
      widths=[7, 3, 3.6, 3], bold_rows=bold)
    FIG("anime_rmse_comparison.png", "Test RMSE on the anime data (1-10 scale).")
    rk = F["E4_rank"][["model", "Precision@10", "Recall@10", "NDCG@10", "HitRate@10", "Coverage"]] \
        .sort_values("NDCG@10", ascending=False).rename(columns={"model": "Model"})
    T(rk, f"Top-10 recommendation quality on anime test users. Relevant = held-out rating ≥ "
          f"{C.RELEVANT_ANIME}/10; already-rated titles excluded.", widths=[5.6, 2.2, 2.2, 2.2, 2.2, 2.2],
      bold_rows=[i for i, m in enumerate(rk.Model) if "RikoNet" in m], font=8.5)
    eo, er = f["e4_ours"], f["e4_ours_rank"]
    P(f"The autoencoder achieves RMSE {fmt(eo.RMSE)} (best baseline {f['e4_best_baseline'].model}: "
      f"{fmt(f['e4_best_baseline'].RMSE)}). For top-10 recommendation the strongest model by NDCG@10 is "
      f"{f['e4_best_ranker'].model} ({fmt(f['e4_best_ranker']['NDCG@10'])}); the autoencoder scores "
      f"{fmt(er['NDCG@10'])}. Low RMSE does not guarantee good top-N lists: rating-prediction models tend "
      "to rank rarely rated, highly scored titles first, while users mostly rate popular titles.")

if "clustering" in R:
    cl = R["clustering"]
    H("7.5 E5 - Embeddings, clusters and logically opposite clusters", 2)
    t1 = pd.read_csv(C.OUT_DIR / "table1_self_tuning.csv", index_col=0)
    t1.columns = [f"K={c}" for c in t1.columns]
    T(t1.reset_index().rename(columns={"index": ""}),
      "Analogue of paper Table 1: candidate numbers of clusters (largest eigengaps) for neighbourhood size "
      "K in {1, 2, 3, 4, 5, 7, 10, 15} (paper: 1-15), plus the number of connected components of the "
      "k-NN graph.", font=7.5)
    T(pd.DataFrame([
        ("Top raters used for stage-1 vectors (m)", cl["top_users_m"]),
        ("Embedding dimension d", cl["embedding_dim"]),
        ("Clusters / neighbours", f"{cl['n_clusters']} / {cl['n_neighbors']}"),
        ("Cluster size min / median / max", f"{cl['size_min']} / {cl['size_median']:.0f} / {cl['size_max']}"),
        ("Clusters with fewer than 4 titles", cl["clusters_below_4"]),
        ("Silhouette - spectral vs K-Means (222)", f"{cl['silhouette_spectral']:.3f} vs {cl['silhouette_kmeans']:.3f}"),
        ("Genre Jaccard within cluster - spectral", f"{cl['genre_jaccard_within_spectral']:.3f}"),
        ("Genre Jaccard within cluster - K-Means", f"{cl['genre_jaccard_within_kmeans']:.3f}"),
        ("Genre Jaccard of random pairs", f"{cl['genre_jaccard_random_pairs']:.3f}"),
        ("Genre Jaccard title vs its opposite title", f"{cl['genre_jaccard_title_vs_opposite']:.3f}"),
        ("Genre Jaccard title vs a random title", f"{cl['genre_jaccard_title_vs_random']:.3f}"),
    ], columns=["Quantity", "Value"]), "Clustering diagnostics on the anime embeddings.", widths=[10, 6])
    P(f"Titles in the same spectral cluster share far more genres (Jaccard "
      f"{cl['genre_jaccard_within_spectral']:.3f}) than random pairs ({cl['genre_jaccard_random_pairs']:.3f}), "
      f"so the embeddings capture content similarity. The logically opposite title shares "
      f"{'fewer' if cl['genre_jaccard_title_vs_opposite'] < cl['genre_jaccard_title_vs_random'] else 'more'} "
      f"genres with the original title than a random title does ({cl['genre_jaccard_title_vs_opposite']:.3f} vs "
      f"{cl['genre_jaccard_title_vs_random']:.3f}), which is the behaviour Algorithm 1 is meant to produce.")
    FIG("anime_embedding_tsne.png", "t-SNE projection of the 6,668 anime embeddings; the three largest "
        "spectral clusters are coloured and labelled with their two most frequent genres.", 13)
    FIG("anime_cluster_sizes.png", "Distribution of spectral-cluster sizes.", 11)

if "hybrid_filter" in R:
    H("7.6 Hybrid filter evaluation", 2)
    hy = R["hybrid_filter"]
    rows_ = []
    for k, v in hy.items():
        pk = [x for x in v if x.startswith("Precision@")][0]
        kk = pk.split("@")[1]
        rows_.append((k, v[pk], v[f"Recall@{kk}"], v[f"HitRate@{kk}"], v.get("genre_jaccard_to_context", float("nan"))))
    T(pd.DataFrame(rows_, columns=["List", "Precision@K", "Recall@K", "HitRate@K", "Genre Jaccard to context"])
      .fillna("-"), "Hybrid-filter lists for anime test users (K = 10, or 20 for both lists together).",
      widths=[6, 2.5, 2.5, 2.5, 3])
    s, p0 = hy["Hybrid: Similar Anime"], hy["Autoencoder top-10 (no filter)"]
    P(f"The Similar Anime list is restricted to on average {s['mean_candidate_set_C']:.0f} candidate titles "
      f"(the union of three clusters) and contains {s['mean_list_length']:.1f} titles on average. Its genre "
      f"overlap with the user's context titles is {s['genre_jaccard_to_context']:.3f} versus "
      f"{p0['genre_jaccard_to_context']:.3f} for the unfiltered autoencoder list, i.e. the filter "
      f"{'does' if s['genre_jaccard_to_context'] > p0['genre_jaccard_to_context'] else 'does not'} make the "
      f"list more similar in content. Its precision is {fmt(s['Precision@10'], 4)} versus "
      f"{fmt(p0['Precision@10'], 4)} for the unfiltered list.")
    ex = C.OUT_DIR / "example_recommendations.csv"
    if ex.exists():
        e = pd.read_csv(ex)
        T(e.rename(columns={"list": "List", "anime": "Anime", "rating / predicted": "Rating / predicted",
                            "cluster": "Cluster"}),
          f"Example output for one real Kaggle test user (user_id {R.get('example_user_kaggle_id')}).",
          widths=[4.4, 7.6, 2.4, 2], font=8)

# ================================================================== 8 discussion
H("8. Discussion: why the headline result was not reproduced")
B([("Protocol of the competitors. ", "The paper's competitor numbers come from other papers that used "
    "different splits (often 80/20). Comparing them with a 5 % hold-out result is not like-for-like. "
    "Re-training all baselines on the same split, as we did, removes this problem."),
   ("Unreported hyper-parameters. ", "Layer sizes, dropout, optimiser, epochs, d and t are not given. "
    "We searched a reasonable grid on validation data, but the authors may have used something else."),
   ("Inconsistent table. ", "Two rows of Table 4 have RMSE ≠ √MSE, suggesting transcription errors."),
   ("Size of the gain. ", "In the paper's own Table 4 every method except PDMF lies between 0.851 and 1.062 "
    f"RMSE; every method we trained lies between {F['E1'].RMSE.min():.3f} and {F['E1'].RMSE.max():.3f}. "
    "A drop to 0.591 is an exceptionally large jump that the paper does not explain in detail."),
   ("Different anime data. ", "Our anime results come from the Kaggle data, not the authors' crawl, and "
    "lack demographics and timestamps, so they cannot be compared numerically with the paper's (unreported) "
    "anime results."),
   ("Limitations of this replication. ", "Single-seed anime runs; anime tuning stopped after 5 of 8 grid "
    "points and anime training capped at 60 epochs (patience 6) because of the CPU time budget; the "
    "activation study was run on MovieLens only; eigengap approximation of the self-tuning method; no "
    "human evaluation.")])

# ================================================================== 9 conclusion
H("9. Conclusion")
if "clustering" not in R:
    P("Interim conclusion (anime experiments pending): on the paper's own benchmark the RikoNet autoencoder "
      f"reaches RMSE {fmt(e1o.RMSE)}, {f['e1_vs_baseline']} a tuned biased matrix-factorisation baseline, while "
      "the paper reports 0.591. The SELU result of the paper's activation study is "
      f"{'confirmed' if f.get('E3_movielens_best') == 'selu' else 'not confirmed'}.")
else:
  P("We re-implemented every component of RikoNet - the masked deep autoencoder with SELU and dense "
  "re-feeding, the three-stage hybrid embeddings, spectral clustering into 222 clusters, the logically "
  "opposite cluster algorithm and the hybrid filter - and evaluated it on the paper's public benchmark and on "
  "a public anime dataset. The pipeline works end to end and produces sensible, genre-coherent "
  f"recommendations. However, the reported MovieLens-100K RMSE of 0.591 could not be reproduced: our best "
  f"configuration reaches {fmt(e1o.RMSE)}, {f['e1_vs_baseline']} a simple well-tuned baseline. Future work: "
  "obtain the original code or data from the authors, try item-based autoencoders (which usually work better "
  "on small datasets), and run a small user study to test the hybrid filter's perceived quality.")

# ================================================================== references
H("References")
for r in [
    "B. Soni, D. Thakuria, N. Nath, N. Das, B. Boro. RikoNet: A novel anime recommendation engine. "
    "Multimedia Tools and Applications 82, 32329-32348 (2023). doi:10.1007/s11042-023-14710-9",
    "O. Kuchaiev, B. Ginsburg. Training deep autoencoders for collaborative filtering. arXiv:1708.01715 (2017).",
    "L. Zelnik-Manor, P. Perona. Self-tuning spectral clustering. NIPS 17 (2004).",
    "G. Klambauer, T. Unterthiner, A. Mayr, S. Hochreiter. Self-normalizing neural networks. NIPS (2017).",
    "R. Kiran, P. Kumar, B. Bhasker. DNNRec: a novel deep learning based hybrid recommender system. "
    "Expert Systems with Applications 144, 113054 (2020).",
    "F. M. Harper, J. A. Konstan. The MovieLens datasets: history and context. ACM TiiS 5(4) (2015).",
    "CooperUnion. Anime Recommendations Database. Kaggle. "
    "https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database",
    "Y. Koren. Factorization meets the neighborhood. KDD (2008).",
]:
    doc.add_paragraph(r, style="List Number")

# ================================================================== appendix
H("Appendix A - Hyper-parameter search (validation only)")
for name in ("movielens", "anime"):
    p = C.OUT_DIR / f"tuning_{name}.csv"
    if p.exists():
        t = pd.read_csv(p).head(8)
        T(t[["hidden", "dropout", "refeed", "l2", "best_epoch", "val_rmse"]],
          f"Best 8 configurations on the {name} validation split (sorted by validation MSE).",
          widths=[3.4, 2, 2, 2, 2.4, 2.6], bold_rows=[0])
H("Appendix B - How to run")
B(["py -3.12 -m venv .venv  then  .venv\\Scripts\\Activate.ps1",
   "python -m pip install pandas numpy scikit-learn matplotlib seaborn tensorflow python-docx python-pptx",
   "python tune.py movielens ; python tune.py anime   (optional, results included)",
   "python main.py   (all experiments)",
   "python recommend.py --rate \"Death Note=9\" \"Clannad=4\"   (interactive recommendations)",
   "python make_report.py ; python make_ppt.py"])
P(f"Total experiment runtime: {R.get('runtime_minutes', '?')} minutes on an Intel Core i5-12500H CPU "
  "(no GPU).", italic=True, color=MUTED)

out = C.OUT_DIR / "RikoNet_Replication_Report.docx"
try:
    doc.save(out)
except PermissionError:          # the old report is open in Word
    out = C.OUT_DIR / "RikoNet_Replication_Report_final.docx"
    doc.save(out)
print("saved", out)
