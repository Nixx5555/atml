"""
data_prep.py - loading, cleaning and splitting the two datasets.

1. MovieLens 100K   - the benchmark the paper itself reports (Table 4). Public, identical data.
2. Kaggle "Anime Recommendations Database" (CooperUnion) - ALTERNATIVE to the paper's private
   MyAnimeList crawl, which the authors never released.

Both datasets end up in the same `RatingData` object so the model / evaluation code is shared.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp

import config as C


@dataclass
class RatingData:
    name: str
    ratings: pd.DataFrame          # columns: u, i, r, order, split  (u, i are 0-based indices)
    n_users: int
    n_items: int
    item_names: list
    item_genres: np.ndarray        # (n_items, n_genres) 0/1 matrix
    genre_names: list
    r_min: float
    r_max: float
    side: np.ndarray | None = None  # (n_users, s) demographic features, or None
    side_names: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    # ---- sparse user x item matrices for each split -------------------------------------
    def matrix(self, *splits) -> sp.csr_matrix:
        d = self.ratings[self.ratings.split.isin(splits)]
        return sp.csr_matrix((d.r.values.astype(np.float32), (d.u.values, d.i.values)),
                             shape=(self.n_users, self.n_items))


# ============================================================================ splitting
def split_per_user(df: pd.DataFrame, test_frac: float, val_frac: float, seed: int) -> pd.Series:
    """Hide a random `test_frac` (and `val_frac`) of EACH user's ratings.

    This is exactly the paper's protocol (Sec. 4.1.3: "5% of the non-zero ratings of each user are
    randomly converted to 0"). The hidden ratings are never shown to any model during training, so
    there is no leakage. The validation part is only used for early stopping / tuning so the test
    part stays untouched until the final evaluation.
    """
    rng = np.random.default_rng(seed)
    rank = pd.Series(rng.random(len(df)), index=df.index).groupby(df.u).rank(method="first") - 1
    n = df.groupby("u").u.transform("size")
    n_test = np.maximum(1, np.round(n * test_frac)).astype(int)
    n_val = np.maximum(1, np.round(n * val_frac)).astype(int) if val_frac > 0 else 0
    split = np.where(rank < n_test, "test", np.where(rank < n_test + n_val, "val", "train"))
    return pd.Series(split, index=df.index)


# ============================================================================ MovieLens 100K
def age_category(age):
    """Paper Sec. 4.1.2 age buckets: <11, 11-15, 16-19, 20-26, >=27  -> 1..5"""
    return np.select([age < 11, age < 16, age < 20, age < 27], [1, 2, 3, 4], 5)


def load_movielens(test_frac=C.TEST_FRACTION, val_frac=C.VAL_FRACTION, seed=C.SEED) -> RatingData:
    d = C.DATA_DIR / "ml-100k"
    df = pd.read_csv(d / "u.data", sep="\t", names=["user", "item", "r", "ts"])
    users = pd.read_csv(d / "u.user", sep="|", names=["user", "age", "gender", "occ", "zip"])
    genre_names = [g.split("|")[0] for g in (d / "u.genre").read_text().strip().splitlines()]
    items = pd.read_csv(d / "u.item", sep="|", encoding="latin-1", header=None)

    df["u"] = df.user - 1
    df["i"] = df.item - 1
    df["order"] = df.ts                       # MovieLens has real timestamps
    df = df[["u", "i", "r", "order"]].copy()
    df["split"] = split_per_user(df, test_frac, val_frac, seed)

    # demographic side-information, encoded as in the paper (Table 3): gender (M=0, F=1) and
    # one-hot age categories 2..5 (category 1 is the all-zero reference)
    users = users.sort_values("user")
    cat = age_category(users.age.values)
    side = np.column_stack([(users.gender == "F").astype(np.float32)] +
                           [(cat == k).astype(np.float32) for k in (2, 3, 4, 5)])

    return RatingData(
        name="MovieLens-100K", ratings=df, n_users=943, n_items=1682,
        item_names=items[1].tolist(), item_genres=items.iloc[:, 5:].values.astype(np.float32),
        genre_names=genre_names, r_min=1, r_max=5, side=side,
        side_names=["gender", "age_cat2", "age_cat3", "age_cat4", "age_cat5"],
        stats=dict(users=943, items=1682, ratings=len(df), density=len(df) / (943 * 1682)))


# ============================================================================ Kaggle anime
def load_anime(test_frac=C.TEST_FRACTION, val_frac=C.VAL_FRACTION, seed=C.SEED,
               n_users=C.N_USERS_ANIME, n_items=C.N_ANIME) -> RatingData:
    anime = pd.read_csv(C.DATA_DIR / "anime.csv")
    rating = pd.read_csv(C.DATA_DIR / "rating.csv")
    stats = dict(raw_anime=len(anime), raw_rows=len(rating))

    # --- cleaning, mirroring Sec. 4.1.2 as far as the Kaggle columns allow ------------------
    # (a) paper drops anime with unknown studio/source; Kaggle has no such columns, so we drop
    #     anime whose genre or type is unknown (the closest available equivalent).
    anime = anime.dropna(subset=["genre", "type"])
    # (b) -1 means "watched but not scored" -> not an explicit rating, discard (like the paper's
    #     discarded "plan-to-watch" pseudo ratings).
    stats["unscored_rows"] = int((rating.rating == -1).sum())
    rating = rating[(rating.rating > 0) & rating.anime_id.isin(anime.anime_id)]
    # (c) duplicates: keep the last entry
    stats["duplicates"] = int(rating.duplicated(["user_id", "anime_id"]).sum())
    rating = rating.drop_duplicates(["user_id", "anime_id"], keep="last")
    # (d) size reduction to the paper's numbers: most-rated 6,668 anime, then the 16,384 most
    #     active users (paper: "removing all users who have rated less than [a threshold]")
    keep_items = rating.anime_id.value_counts().index[:n_items]
    rating = rating[rating.anime_id.isin(keep_items)]
    counts = rating.user_id.value_counts()
    keep_users = counts.index[:n_users]
    stats["user_min_ratings"] = int(counts.iloc[n_users - 1])
    rating = rating[rating.user_id.isin(keep_users)].copy()

    # --- index mapping ----------------------------------------------------------------------
    item_ids = np.sort(rating.anime_id.unique())
    user_ids = np.sort(rating.user_id.unique())
    rating["u"] = np.searchsorted(user_ids, rating.user_id.values)
    rating["i"] = np.searchsorted(item_ids, rating.anime_id.values)
    rating["r"] = rating.rating
    # Kaggle has no timestamps -> file order is used as a stand-in for "temporal order"
    rating["order"] = np.arange(len(rating))
    df = rating[["u", "i", "r", "order"]].reset_index(drop=True)
    df["split"] = split_per_user(df, test_frac, val_frac, seed)

    # --- anime-genre sparse matrix (paper Table 2) -------------------------------------------
    anime = anime.set_index("anime_id").loc[item_ids]
    genre_lists = anime.genre.str.split(r",\s*")
    genre_names = sorted({g for gl in genre_lists for g in gl})
    G = np.zeros((len(item_ids), len(genre_names)), dtype=np.float32)
    gidx = {g: k for k, g in enumerate(genre_names)}
    for row, gl in enumerate(genre_lists):
        G[row, [gidx[g] for g in gl]] = 1

    stats.update(users=len(user_ids), items=len(item_ids), ratings=len(df), genres=len(genre_names),
                 density=len(df) / (len(user_ids) * len(item_ids)), mean_rating=float(df.r.mean()))
    data = RatingData(name="Anime (Kaggle)", ratings=df, n_users=len(user_ids),
                      n_items=len(item_ids), item_names=anime.name.tolist(), item_genres=G,
                      genre_names=genre_names, r_min=1, r_max=10, stats=stats)
    data.item_ids = item_ids
    data.user_ids = user_ids
    data.item_members = anime.members.values
    return data
