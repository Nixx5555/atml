"""
demo.py - 2-minute live demonstration for the viva. Writes NOTHING to outputs/.

    python demo.py

1. loads MovieLens-100K and hides 5 % of every user's ratings (paper protocol)
2. trains the RikoNet autoencoder (validation-tuned config) with early stopping
3. compares its test RMSE / MAE with three baselines trained on the same split
"""
import json
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np

import baselines as B
import config as C
import data_prep as D
import evaluate as E
from model import DeepAutoencoder

data = D.load_movielens()
Xtr, Xv, Xte = data.matrix("train"), data.matrix("val"), data.matrix("test")
print(f"MovieLens-100K: {data.n_users} users x {data.n_items} movies | "
      f"train {Xtr.nnz:,}  val {Xv.nnz:,}  test {Xte.nnz:,} ratings")

p = C.OUT_DIR / "best_config_movielens.json"
cfg = json.loads(p.read_text()) if p.exists() else dict(hidden=[512, 512, 1024], dropout=0.8,
                                                        refeed=True, l2=0.0)
print("RikoNet autoencoder config:", cfg)
t0 = time.time()
ae = DeepAutoencoder(data.n_items, n_side=data.side.shape[1], hidden=tuple(cfg["hidden"]),
                     dropout=cfg["dropout"], refeed=cfg["refeed"], l2=cfg["l2"])
h = ae.fit(Xtr, Xv, data.side, label="demo")
print(f"trained in {time.time() - t0:.0f}s, best epoch {h['best_epoch']}\n")

rows = [("RikoNet autoencoder", E.rating_metrics(lambda r: ae.predict(Xtr, data.side, r), Xte, 1, 5))]
for bl in (B.GlobalMean(), B.UserMean(), B.BiasBaseline()):
    bl.fit(Xtr)
    rows.append((bl.name, E.rating_metrics(bl.predict_rows, Xte, 1, 5)))

print(f"{'model':32s} {'RMSE':>7s} {'MAE':>7s}")
for name, m in sorted(rows, key=lambda x: x[1]["RMSE"]):
    print(f"{name:32s} {m['RMSE']:7.4f} {m['MAE']:7.4f}")
print(f"\nPaper reports RikoNet RMSE 0.591 on this dataset.")
