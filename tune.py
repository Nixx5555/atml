"""
tune.py - choose the autoencoder hyper-parameters the paper does NOT report.

Selection uses ONLY the validation split (5 % of each user's ratings, disjoint from the test
split). The test split is never read here, so the final test numbers are not optimistically biased.

Usage:  python tune.py movielens      (about 10 minutes on a laptop CPU)
        python tune.py anime          (8 configurations, about 40-60 minutes)
Writes: outputs/tuning_<dataset>.csv  and  outputs/best_config_<dataset>.json
"""
import itertools
import json
import os
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import pandas as pd

import config as C
import data_prep as D
from model import DeepAutoencoder

GRIDS = {
    "movielens": dict(hidden=[(512, 512, 1024), (256, 256, 512), (512,), (256, 128)],
                      dropout=[0.8, 0.5, 0.2], refeed=[True, False], l2=[0.0, 1e-4]),
    "anime": dict(hidden=[(512, 512, 1024), (256, 256, 512)], dropout=[0.8, 0.5],
                  refeed=[True, False], l2=[1e-4]),
}


def main(name):
    data = D.load_movielens() if name == "movielens" else D.load_anime()
    Xtr, Xv = data.matrix("train"), data.matrix("val")
    side = data.side
    rows = []
    grid = GRIDS[name]
    for hidden, dropout, refeed, l2 in itertools.product(*grid.values()):
        t0 = time.time()
        ae = DeepAutoencoder(data.n_items, n_side=0 if side is None else side.shape[1],
                             hidden=hidden, dropout=dropout, refeed=refeed, l2=l2)
        h = ae.fit(Xtr, Xv, side, verbose=False, label=name)
        row = dict(hidden="-".join(map(str, hidden)), dropout=dropout, refeed=refeed, l2=l2,
                   best_epoch=h["best_epoch"], val_mse=min(h["val_mse"]),
                   val_rmse=float(np.sqrt(min(h["val_mse"]))), seconds=round(time.time() - t0))
        rows.append(row)
        print(row, flush=True)
    df = pd.DataFrame(rows).sort_values("val_mse")
    df.to_csv(C.OUT_DIR / f"tuning_{name}.csv", index=False)
    best = df.iloc[0]
    cfg = dict(hidden=[int(x) for x in best.hidden.split("-")], dropout=float(best.dropout),
               refeed=bool(best.refeed), l2=float(best.l2))
    (C.OUT_DIR / f"best_config_{name}.json").write_text(json.dumps(cfg, indent=2))
    print("best:", cfg)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "movielens")
