"""
config.py - every tunable setting of the replication in one place.

Each value is tagged:
  [PAPER]  taken directly from the RikoNet paper (Soni et al., MTAP 2023)
  [K&G]    taken from Kuchaiev & Ginsburg (2017), the autoencoder RikoNet says it is based on
  [OURS]   not reported in the paper, so we chose it (explain this in the viva!)
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
MODEL_DIR = OUT_DIR / "models"

SEED = 42

# ---------------------------------------------------------------- data
N_USERS_ANIME = 16_384        # [PAPER] users kept after filtering
N_ANIME = 6_668               # [PAPER] anime titles kept after cleaning
TEST_FRACTION = 0.05          # [PAPER] 5 % of each user's ratings hidden for testing
VAL_FRACTION = 0.05           # [OURS] another 5 % hidden for early stopping (keeps test set untouched)
STANDARD_TEST_FRACTION = 0.20 # [OURS] extra experiment: the more common 80/20 protocol

# ---------------------------------------------------------------- primary autoencoder
HIDDEN = (512, 512, 1024)     # [K&G] n,512,512,1024,dp(0.8),512,512,n
ACTIVATION = "selu"           # [PAPER] SELU in hidden layers
FINAL_ACTIVATION = "relu"     # [PAPER] ReLU on the output layer
DROPOUT = 0.8                 # [K&G] dropout on the code layer
DENSE_REFEEDING = True        # [K&G] iterative output re-feeding
LEARNING_RATE = 1e-3          # [OURS] Adam (paper does not name an optimiser)
BATCH_SIZE = 128              # [K&G]
MAX_EPOCHS = 200              # [OURS] upper bound; early stopping decides
PATIENCE = 15                 # [OURS] epochs without validation improvement before stopping
ACTIVATIONS_TO_COMPARE = ("relu", "tanh", "elu", "selu")  # [PAPER] Fig. 3
ACTIVATION_STUDY_EPOCHS = 40  # [OURS]

# ---------------------------------------------------------------- embeddings + clustering
TOP_USER_MIN_RATINGS = 500    # [OURS] threshold t for "top raters" (paper: value not given)
EMB_HIDDEN = (256, 128, 64)   # [OURS] stage-1 autoencoder, code size d = 64
EMB_DIM = 64                  # [OURS] d (paper: value not given)
HYBRID_HIDDEN = (96, 64)      # [OURS] stage-3 autoencoder d' = d + 43 -> d
N_CLUSTERS = 222              # [PAPER] chosen number of clusters
MIN_CLUSTER_ELEMS = 4         # [PAPER] "minimum elements in a cluster" -> used as neighbour count
SELF_TUNING_RANGE = range(1, 16)  # [PAPER] Table 1 columns 1..15
EPSILON = 1e-9                # [PAPER] Algorithm 1

# ---------------------------------------------------------------- hybrid filter / ranking
LAST_N_RATED = 3              # [PAPER] last 3 rated titles define the user's context
LIKE_THRESHOLD_ANIME = 7      # [OURS] rating >= 7 -> own cluster, else logically opposite cluster
TOP_K = 10                    # [PAPER] both lists have 10 titles
RELEVANT_ANIME = 8            # [OURS] a held-out rating >= 8/10 counts as a relevant hit
RELEVANT_MOVIELENS = 4        # [OURS] a held-out rating >= 4/5 counts as a relevant hit

for d in (OUT_DIR, FIG_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)
