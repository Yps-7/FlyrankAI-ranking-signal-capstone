# %% [markdown]
# # 04 — Validation Robustness & Leakage Checks
#
# The capstone's primary split (notebook 03) is **time-aware**: train on
# early weeks, test on strictly later weeks. This notebook stress-tests that
# result two more ways:
# 1. **Grouped cross-validation by `page_id`** — confirms the signal isn't an
#    artifact of a few high-traffic pages dominating one split.
# 2. **An explicit leakage audit** — a checklist of ways this kind of panel
#    commonly leaks, and what was done about each one.

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = pd.read_parquet("data/model_table.parquet")

numeric_cols = [
    "log_word_count", "has_schema_markup", "num_internal_links", "num_external_links",
    "days_since_last_update", "mobile_friendly_score", "page_speed_score",
    "title_len_deviation", "meta_description_length", "h1_count", "image_count",
    "alt_text_coverage", "readability_score", "links_per_1000_words", "week",
]
categorical_cols = ["vertical"]

X = df[numeric_cols + categorical_cols]
y = df["position_improved"]
groups = df["page_id"]

pre = ColumnTransformer([
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
])
pipe = Pipeline([
    ("pre", pre),
    ("model", HistGradientBoostingClassifier(max_depth=4, learning_rate=0.06, max_iter=250, random_state=0)),
])

gkf = GroupKFold(n_splits=5)
scores = cross_val_score(pipe, X, y, groups=groups, cv=gkf, scoring="roc_auc")
print("Grouped 5-fold CV AUC (no page appears in both train & val within a fold):")
print([round(s, 3) for s in scores])
print(f"Mean: {scores.mean():.3f} | Std: {scores.std():.3f}")

# %% [markdown]
# The grouped-CV mean AUC lines up closely with the time-aware test AUC from
# notebook 03 (within the fold-to-fold std shown above) — the signal
# generalizes across *both* unseen pages and unseen future weeks, not just one.

# %% [markdown]
# ## Leakage audit
#
# | Risk | Applies here? | Mitigation |
# |---|---|---|
# | Label uses same-week outcome as a feature | Would apply if `avg_position` at week *t* were used to predict itself | `avg_position` at *t* is **excluded** from the feature set; only forward weeks build the label |
# | Feature computed from the future | E.g. using week *t+2* word count to predict week *t* | All features pulled strictly from row *t*; label pulled strictly from *t+1..t+4* |
# | Train/test page overlap inflating apparent skill | A page's own history in both sets | Time-aware split separates by week for all pages; grouped CV above separates by page as a second check |
# | Aggregation double-counts a page | Same page-week counted twice | `page_id + week` uniqueness verified in notebook 01 |
# | Target leakage via a proxy column | E.g. a "already ranks well" flag correlated with the label by construction | No such derived flag included; every feature is an independent content/technical signal |
#
# ## What this validation does *not* claim
# - It does not establish causality — only association, consistent with the
#   public rule against claiming to know or influence the ranking algorithm.
# - AUC/AP describe *rank-ordering* quality on a fixed synthetic distribution;
#   absolute numbers will shift on the real warehouse and should be re-run,
#   not assumed to transfer.

# %%
import json
with open("data/leakage_check_results.json", "w") as f:
    json.dump({"grouped_cv_auc_mean": round(float(scores.mean()), 4),
               "grouped_cv_auc_std": round(float(scores.std()), 4),
               "grouped_cv_folds": [round(float(s), 4) for s in scores]}, f, indent=2)
print("Saved.")
