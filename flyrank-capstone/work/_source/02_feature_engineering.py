# %% [markdown]
# # 02 — Feature Engineering & Label Definition
#
# Builds the modeling table for the Ranking Signal Analysis lane:
# **which safe content/technical signals are associated with visibility,
# clicks, engagement, or (forward) movement?**
#
# Two outcomes are engineered:
# 1. `ctr` — already in the panel, used directly for the CTR/engagement signal cut.
# 2. `position_improved` — a **forward-looking, leakage-safe** label: did the
#    page's average position improve by at least 3 spots, comparing the mean
#    of weeks *t+1..t+4* against week *t*? Only rows with a full 4-week future
#    window are labeled; the last 4 weeks of the panel are dropped from
#    training (they have no future to check).

# %%
import duckdb
import pandas as pd
import numpy as np

pd.set_option("display.max_columns", 30)

panel = duckdb.sql("SELECT * FROM read_csv_auto('data/signals_panel.csv')").df()
panel = panel.sort_values(["page_id", "week"]).reset_index(drop=True)

MAX_WEEK = panel["week"].max()
FORWARD_WINDOW = 4
IMPROVEMENT_THRESHOLD = 3.0  # positions

# %% [markdown]
# ## Forward label construction (no leakage: label only uses weeks > t)

# %%
pos_wide = panel.pivot(index="page_id", columns="week", values="avg_position")


def forward_label_for_week(t: int) -> pd.Series:
    future_cols = [w for w in range(t + 1, t + 1 + FORWARD_WINDOW) if w in pos_wide.columns]
    if len(future_cols) < FORWARD_WINDOW:
        return pd.Series(index=pos_wide.index, dtype="float64")  # not enough future -> unlabeled
    future_mean = pos_wide[future_cols].mean(axis=1)
    improved = (pos_wide[t] - future_mean) >= IMPROVEMENT_THRESHOLD  # position number goes DOWN = better
    return improved.astype(float)


labels = {t: forward_label_for_week(t) for t in range(MAX_WEEK + 1)}
label_df = pd.DataFrame(labels)
label_df.columns = [f"w{c}" for c in label_df.columns]
label_long = label_df.reset_index().melt(id_vars="page_id", var_name="week", value_name="position_improved")
label_long["week"] = label_long["week"].str.replace("w", "").astype(int)

# %%
feat = panel.merge(label_long, on=["page_id", "week"], how="left")
n_labeled = feat["position_improved"].notna().sum()
n_dropped = feat["position_improved"].isna().sum()
print(f"Labeled rows: {n_labeled:,} | Dropped (no future window, e.g. last {FORWARD_WINDOW} weeks): {n_dropped:,}")
feat["position_improved"].value_counts(dropna=False, normalize=True).round(3)

# %% [markdown]
# ## Engineered features
#
# All features below are known **as of week t** (nothing from the future).

# %%
feat["freshness_bucket"] = pd.cut(
    feat["days_since_last_update"], bins=[-1, 21, 60, 180, 100000],
    labels=["<=21d", "22-60d", "61-180d", ">180d"],
)
feat["title_len_deviation"] = (feat["title_length"] - 58).abs()  # distance from a common SERP-friendly length
feat["log_word_count"] = np.log1p(feat["word_count"])
feat["links_per_1000_words"] = feat["num_internal_links"] / (feat["word_count"] / 1000)
feat["vertical"] = feat["vertical"].astype("category")

feature_cols = [
    "log_word_count", "has_schema_markup", "num_internal_links", "num_external_links",
    "days_since_last_update", "mobile_friendly_score", "page_speed_score",
    "title_len_deviation", "meta_description_length", "h1_count", "image_count",
    "alt_text_coverage", "readability_score", "links_per_1000_words", "vertical",
]
model_df = feat.dropna(subset=["position_improved"]).copy()
model_df = model_df[feature_cols + ["page_id", "week", "avg_position", "ctr", "position_improved"]]
model_df.to_parquet("data/model_table.parquet", index=False)
print(model_df.shape)
model_df.head()

# %% [markdown]
# ## Exclusions log (public-safe)
# - Rows in the last 4 weeks of the panel (25,900 → excluded from labeled
#   table): no valid forward window, would otherwise leak an incomplete
#   future into training.
# - No pages excluded for missingness (none present, see notebook 01).
# - No query-level or click-level raw exports used anywhere downstream —
#   only page-week aggregates.

# %% [markdown]
# ## Quick signal cuts (descriptive, pre-model)

# %%
schema_ctr = model_df.groupby("has_schema_markup")["ctr"].mean().round(4)
print("Mean CTR by schema markup presence:\n", schema_ctr)

fresh_lookup = feat.loc[model_df.index, "freshness_bucket"]
fresh_improve = (
    model_df.assign(freshness_bucket=fresh_lookup)
    .groupby("freshness_bucket", observed=True)["position_improved"]
    .mean()
    .round(4)
)
print("\nP(position improved) by freshness bucket:\n", fresh_improve)
