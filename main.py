# ============================================================
# FORECAST AGGREGATION AND PERFORMANCE EVALUATION PIPELINE
# ============================================================

import time
import numpy as np #used for faster calculations, especially for the aggregation methods
import pandas as pd

import warnings
warnings.filterwarnings("ignore")# Suppress harmless warnings for cleaner output


# ------------------------------------------------------------
# 1) LOAD DATA
# ------------------------------------------------------------
print("Loading data...")

#Paths, this script assumes all the working data is in the working directory.
path_prediction_sets = "rct-a-prediction-sets.csv"
path_daily_forecasts = "rct-a-daily-forecasts.csv"
path_qanda = "rct-a-questions-answers.csv"

prediction_sets = pd.read_csv(path_prediction_sets)
daily_forecasts = pd.read_csv(path_daily_forecasts)
qanda = pd.read_csv(path_qanda)

print(f"Loaded:")
print(f"  • Prediction sets: {len(prediction_sets):,}")
print(f"  • Daily forecasts: {len(daily_forecasts):,}")
print(f"  • Q&A metadata:    {len(qanda):,}\n")

# ------------------------------------------------------------
# 2) BASIC CLEANING
# ------------------------------------------------------------
# Keep relevant columns only
df = daily_forecasts[
    ["external prediction set id", "external predictor id",
     "discover question id", "discover answer id", "forecast", "date"]
].copy()

# Parse timestamps and ensure numeric forecasts
df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
df["forecast"] = pd.to_numeric(df["forecast"], errors="coerce")

# Drop invalid rows
df = df.dropna(subset=["date", "forecast"])

# Convert IDs to category to save memory
for col in ["external prediction set id", "external predictor id",
            "discover question id", "discover answer id"]:
    df[col] = df[col].astype("category")

print(f"After cleaning: {len(df):,} valid forecast records.\n")

# ------------------------------------------------------------
# 3) REMOVE STALE/SYSTEM DEFAULT PREDICTION SETS
# ------------------------------------------------------------
# A "stale" prediction set is one where all answers have identical forecasts. 
# These essentially imply a system default, i.e no forecast was made.
 
rounded = df["forecast"].round(6)
is_stale = df.groupby("external prediction set id", observed=True)[rounded.name].transform("nunique") <= 1
df_filtered = df.loc[~is_stale].copy()

print(f"Removed {is_stale.sum():,} stale forecasts (no updates within a set).\n")

# ------------------------------------------------------------
# 4) AGGREGATE FORECASTS BY (DATE, QUESTION, ANSWER)
# ------------------------------------------------------------
# Combine all individual forecasts for each question-answer-day
# using the specified aggregation techniques

EPS = 1e-12  # avoids log(0) in geometric means

# Helper functions for aggregation (NumPy-based for speed)
def geometric_mean(x):
    a = np.clip(x.to_numpy(), EPS, 1 - EPS)
    return np.exp(np.mean(np.log(a)))

def geometric_mean_of_odds(x):
    a = np.clip(x.to_numpy(), EPS, 1 - EPS)
    odds = a / (1 - a)
    gm_odds = np.exp(np.mean(np.log(odds)))
    return gm_odds / (1 + gm_odds)

def trimmed_mean(x, trim_frac=0.1):
    a = np.sort(x.to_numpy())
    n = len(a)
    if n == 0:
        return np.nan
    k = int(np.floor(trim_frac * n))
    return a[k:n-k].mean() if n > 2 * k else a.mean()

print("Aggregating forecasts...")
t0 = time.time()

aggregated = (
    df_filtered.groupby(["date", "discover question id", "discover answer id"], observed=True, sort=False)["forecast"]
    .agg(
        raw_mean="mean",
        median="median",
        geometric_mean=geometric_mean,
        trimmed_mean=lambda s: trimmed_mean(s, 0.1),
        geometric_mean_of_odds=geometric_mean_of_odds,
    )
    .reset_index()
)

print(f"Aggregation completed in {time.time() - t0:.2f}s. Sample:")
print(aggregated.head(), "\n")


# ------------------------------------------------------------
# 5) MERGE QUESTION METADATA
# ------------------------------------------------------------
# Add resolved outcome probabilities and ordinal scoring info from the Q&A file.

resolved = qanda[
    ["discover question id", "discover answer id", "answer resolved probability"]
].drop_duplicates()

ordinal_flag = qanda[["discover question id", "use ordinal scoring"]].drop_duplicates()

agg = (
    aggregated.merge(resolved, on=["discover question id", "discover answer id"], how="left")
    .merge(ordinal_flag, on="discover question id", how="left")
)

print("Merged resolved probabilities and scoring flags.\n")

# ------------------------------------------------------------
# 6) BASE AGGREGATION PERFORMANCE (Brier Scores)
# ------------------------------------------------------------
# Evaluate accuracy of each aggregation method using the Brier score.
# For ordinal questions, cumulative (ordered) Brier scoring is used.

def brier_unordered(f, o):
    return np.mean((f - o) ** 2)

def brier_ordered(f, o):
    return np.mean((np.cumsum(f) - np.cumsum(o)) ** 2)

def compute_brier_scores(group):
    """Compute Brier score for each aggregation method."""
    outcomes = group["answer resolved probability"].values
    ordered = group["use ordinal scoring"].iloc[0] is True
    scores = {}
    for m in ["raw_mean", "median", "geometric_mean", "trimmed_mean", "geometric_mean_of_odds"]:
        preds = group[m].values
        scores[m + "_brier"] = (
            brier_ordered(preds, outcomes) if ordered else brier_unordered(preds, outcomes)
        )
    return pd.Series(scores)

print("Calculating baseline Brier scores...")
t0 = time.time()
brier = agg.groupby(["date", "discover question id"], observed=True).apply(compute_brier_scores).reset_index()
agg = agg.merge(brier, on=["date", "discover question id"], how="left")
print(f"Baseline Brier scores computed in {time.time() - t0:.2f}s.\n")

# ------------------------------------------------------------
# 7) IMPROVED METHOD — FORECASTER-SKILL WEIGHTED MEAN 
# ------------------------------------------------------------
# Leak-safe: each forecast at date T uses only forecaster performance from questions
# resolved before T. Faster: vectorized binary search + vectorized weighted mean.
# More accurate/robust: Bayesian shrinkage toward global resolved mean + weight caps.

print("Computing forecaster-skill-weighted mean...")
t7 = time.time()

# ---- Config knobs----
EPS = 1e-6               # numerical floor
PRIOR_STRENGTH = 10      # pseudo-counts for shrinkage toward global mean Brier
MAX_WEIGHT_MULT = 50.0   # cap: max relative weight vs average weight per (date, q, a)

# ---- 7.1 Prepare resolution timestamp ----
qanda["question_correctness_known_at"] = pd.to_datetime(
    qanda["question correctness known at"].str.replace(" UTC", "", regex=False),
    utc=True,
    errors="coerce"
)

# ---- 7.2 Merge forecasts with resolution metadata; compute per-row Brier on resolved ----
merged = df.merge(
    qanda[[
        "discover question id",
        "discover answer id",
        "answer resolved probability",
        "question_correctness_known_at",
    ]],
    on=["discover question id", "discover answer id"],
    how="left",
)

resolved = merged[merged["question_correctness_known_at"].notna()].copy()
resolved["brier_score"] = (resolved["forecast"] - resolved["answer resolved probability"]) ** 2

# ---- Ensure UTC & stable dtypes ----
df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
resolved["date"] = pd.to_datetime(resolved["date"], utc=True, errors="coerce")
df["external predictor id"] = df["external predictor id"].astype(str)
resolved["external predictor id"] = resolved["external predictor id"].astype(str)

# ---- 7.3 Build per-forecaster history arrays (sorted by correctness-known time) ----
resolved = resolved.sort_values(
    ["external predictor id", "question_correctness_known_at"]
).reset_index(drop=True)

# cumulative sums & counts → fast means and shrinkage
resolved["_cum_sum"] = (
    resolved.groupby("external predictor id", observed=True)["brier_score"].cumsum()
)
resolved["_cum_cnt"] = (
    resolved.groupby("external predictor id", observed=True)["brier_score"].cumcount() + 1
)

# global mean for shrinkage fallback
global_mean_brier = resolved["brier_score"].mean()

# map forecaster → arrays of (known_time, cum_sum, cum_cnt)
hist_map = {
    fid: grp[["question_correctness_known_at", "_cum_sum", "_cum_cnt"]].to_numpy()
    for fid, grp in resolved.groupby("external predictor id")
}

# ---- 7.4 Vectorized historical skill lookup (binary search per forecaster) ----
print("  • Assigning historical skill per forecaster...")
t_lookup = time.time()

skills = np.empty(len(df), dtype=float)
skills[:] = np.nan

# pre-extract columns as arrays for speed
df_ids = df["external predictor id"].to_numpy()
df_dates = df["date"].values.astype("datetime64[ns]")

# group indices once to avoid pandas overhead in a loop
# (np.where per-id is fast because we use boolean masks on arrays)
unique_ids, inverse = np.unique(df_ids, return_inverse=True)

for i, fid in enumerate(unique_ids):
    idx = (inverse == i)
    if fid not in hist_map:
        continue  # stays NaN → will be filled with global mean
    hist = hist_map[fid]
    known_times = hist[:, 0].astype("datetime64[ns]")
    cum_sum = hist[:, 1].astype(float)
    cum_cnt = hist[:, 2].astype(float)

    f_dates = df_dates[idx]
    pos = np.searchsorted(known_times, f_dates, side="left") - 1  # strictly before T

    valid = pos >= 0
    # historical mean before T
    h_sum = np.zeros(pos.shape, dtype=float)
    h_cnt = np.zeros(pos.shape, dtype=float)
    h_sum[valid] = cum_sum[pos[valid]]
    h_cnt[valid] = cum_cnt[pos[valid]]

    # Bayesian shrinkage toward global mean to stabilize small histories
    # hist_mean_shrunk = (PRIOR_STRENGTH * global_mean + h_sum) / (PRIOR_STRENGTH + h_cnt)
    hist_mean_shrunk = (PRIOR_STRENGTH * global_mean_brier + h_sum) / np.maximum(PRIOR_STRENGTH + h_cnt, 1.0)

    skills[idx] = np.where(valid, hist_mean_shrunk, np.nan)

# attach and fill any missing with global mean
df["forecaster_mean_brier"] = pd.Series(skills, index=df.index).fillna(global_mean_brier)

# ---- 7.5 Compute weights (inverse of historical mean), capped to avoid domination ----
df["forecaster_weight_raw"] = 1.0 / (df["forecaster_mean_brier"] + EPS)

# normalize & cap per (date, question, answer) to curb any single forecaster dominating
# w_cap = MAX_WEIGHT_MULT * average weight in that group
grp_keys = ["date", "discover question id", "discover answer id"]
mean_w = df.groupby(grp_keys, observed=True)["forecaster_weight_raw"].transform("mean")
cap = MAX_WEIGHT_MULT * (mean_w + EPS)
df["forecaster_weight"] = np.minimum(df["forecaster_weight_raw"], cap)

# ---- 7.6 Vectorized weighted mean: sum(w*p) / sum(w) (no groupby.apply) ----
t_wavg = time.time()
df["wf"] = df["forecaster_weight"] * df["forecast"]

num = df.groupby(grp_keys, observed=True)["wf"].sum()
den = df.groupby(grp_keys, observed=True)["forecaster_weight"].sum()
weighted_mean = (num / (den + EPS)).reset_index(name="forecaster_weighted_mean")

# Merge into agg (ensure UTC dtype)
agg["date"] = pd.to_datetime(agg["date"], utc=True, errors="coerce")
weighted_mean["date"] = pd.to_datetime(weighted_mean["date"], utc=True, errors="coerce")
agg = agg.merge(
    weighted_mean,
    on=["date", "discover question id", "discover answer id"],
    how="left"
)

print(f"  • Skill lookup time: {time.time() - t_lookup:.2f}s")
print(f"  • Weighted-mean time: {time.time() - t_wavg:.2f}s")
print(f"Section 7 total time: {time.time() - t7:.2f}s\n")

# ---- 7.7 Evaluate weighted aggregator via Brier ----
t7b = time.time()
def compute_weighted_brier(group):
    outcomes = group["answer resolved probability"].values
    preds = group["forecaster_weighted_mean"].values
    ordered = group["use ordinal scoring"].iloc[0] is True
    score = brier_ordered(preds, outcomes) if ordered else brier_unordered(preds, outcomes)
    return pd.Series({"forecaster_weighted_brier": score})

brier_weighted = (
    agg.groupby(["date", "discover question id"], observed=True)
       .apply(compute_weighted_brier)
       .reset_index()
)
agg = agg.merge(brier_weighted, on=["date", "discover question id"], how="left")
print(f"Weighted Brier evaluation time: {time.time() - t7b:.2f}s")
print("Forecaster-weighted mean and Brier computed successfully.\n")

# ------------------------------------------------------------
# 8) FINAL SUMMARY — AVERAGE BRIER SCORES
# ------------------------------------------------------------
def summarize_brier(df):
    cols = [
        "raw_mean_brier", "median_brier", "geometric_mean_brier",
        "trimmed_mean_brier", "geometric_mean_of_odds_brier",
        "forecaster_weighted_brier",
    ]
    return (
        df[cols]
        .mean()
        .reset_index()
        .rename(columns={"index": "method", 0: "average_brier"})
        .sort_values("average_brier")
        .reset_index(drop=True)
    )

summary = summarize_brier(agg)
print("Average Brier scores across aggregation methods:\n")
print(summary)
print("\nPipeline executed successfully.")
