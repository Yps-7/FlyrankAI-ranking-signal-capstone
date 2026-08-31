# %% [markdown]
# # 01 — Data Intake & Exploratory Data Analysis
# **Lane:** Ranking Signal Analysis
#
# This notebook loads the Search Intelligence warehouse with DuckDB and profiles
# it before any modeling happens. Right now it points at a **synthetic
# placeholder panel** built to the same schema as the real FlyRank warehouse
# (see `work/data/generate_synthetic_data.py` for exactly what is simulated
# and why). Swap-in instructions for the real gated dataset are in that file's
# docstring — every cell below is written so that swap is the *only* change
# needed.

# %%
import duckdb
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", 30)
plt.rcParams["figure.dpi"] = 110

DATA_PATH = "data/signals_panel.csv"

# %% [markdown]
# ## Load with DuckDB
#
# In production this cell becomes:
# ```python
# con.sql("SET hf_token='<token>';")
# con.sql("SELECT * FROM read_parquet('hf://datasets/<flyrank-repo>/*.parquet')")
# ```
# For now we point DuckDB at the local synthetic CSV so the aggregation
# workflow (and every downstream cell) is identical either way.

# %%
con = duckdb.connect()
panel = con.sql(f"SELECT * FROM read_csv_auto('{DATA_PATH}')").df()
print(f"Rows: {len(panel):,} | Pages: {panel['page_id'].nunique():,} | Weeks: {panel['week'].nunique()}")
panel.head()

# %% [markdown]
# ## Schema & data quality checks

# %%
dtypes = panel.dtypes.to_frame("dtype")
nulls = panel.isna().sum().to_frame("n_null")
quality = dtypes.join(nulls)
quality["pct_null"] = (quality["n_null"] / len(panel) * 100).round(3)
quality

# %%
# Range sanity checks — flag anything outside plausible bounds
checks = {
    "ctr in [0,1]": panel["ctr"].between(0, 1).all(),
    "avg_position in [1,100]": panel["avg_position"].between(1, 100).all(),
    "clicks <= impressions": (panel["clicks"] <= panel["impressions"]).all(),
    "no duplicate page_id+week": not panel.duplicated(subset=["page_id", "week"]).any(),
}
for k, v in checks.items():
    print(f"{'PASS' if v else 'FAIL'} — {k}")

# %% [markdown]
# All checks pass. No rows excluded at this stage — see the capstone
# notebook's **Data** section for the (public-safe) exclusion log applied
# before modeling.

# %% [markdown]
# ## Distributions of key outcomes

# %%
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
panel["avg_position"].hist(bins=40, ax=axes[0], color="#3B6E8F")
axes[0].set_title("Avg. position")
panel["ctr"].hist(bins=40, ax=axes[1], color="#C97B3D")
axes[1].set_title("CTR")
panel.groupby("page_id")["impressions"].sum().pipe(lambda s: s.clip(upper=s.quantile(0.99))).hist(
    bins=40, ax=axes[2], color="#5B7F5A"
)
axes[2].set_title("Total impressions per page (99th pct clipped)")
for ax in axes:
    ax.set_ylabel("pages/rows")
plt.tight_layout()
plt.savefig("../paper/assets/01_outcome_distributions.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Vertical mix and baseline position by vertical

# %%
by_vertical = panel.groupby("vertical").agg(
    pages=("page_id", "nunique"),
    avg_position=("avg_position", "mean"),
    avg_ctr=("ctr", "mean"),
    avg_impressions=("impressions", "mean"),
).round(3).sort_values("avg_position")
by_vertical

# %% [markdown]
# ## Weekly trend — is anything moving in aggregate?

# %%
weekly = panel.groupby("week").agg(avg_position=("avg_position", "mean"), avg_ctr=("ctr", "mean"))
fig, ax1 = plt.subplots(figsize=(9, 3.6))
ax1.plot(weekly.index, weekly["avg_position"], color="#3B6E8F", label="Avg position (lower=better)")
ax1.invert_yaxis()
ax1.set_xlabel("week")
ax1.set_ylabel("Avg position", color="#3B6E8F")
ax2 = ax1.twinx()
ax2.plot(weekly.index, weekly["avg_ctr"], color="#C97B3D", label="Avg CTR")
ax2.set_ylabel("Avg CTR", color="#C97B3D")
plt.title("Warehouse-wide weekly trend")
plt.tight_layout()
plt.savefig("../paper/assets/01_weekly_trend.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Takeaways for feature engineering (notebook 02)
# - Position and CTR both show a mild cyclical wobble — week needs to be
#   available as a feature or the label window needs to be long enough to
#   average it out.
# - Verticals differ enough in baseline position that vertical should be a
#   feature (or the label should be defined relative to the page's own
#   history, not an absolute threshold).
# - No missing data / range violations — no imputation logic needed before
#   feature engineering.
