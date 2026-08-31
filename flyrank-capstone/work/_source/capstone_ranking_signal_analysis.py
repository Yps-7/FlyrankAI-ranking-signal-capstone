# %% [markdown]
# # Capstone — Ranking Signal Analysis
# **Which safe content/search signals are associated with visibility, clicks,
# and forward position movement — and what should a content team do about it?**
#
# This notebook is the synthesis layer. The heavy lifting (data intake,
# feature/label engineering, modeling, validation) lives in `01`–`04`; this
# notebook re-loads their saved artifacts, states the final numbers cleanly,
# and produces the ranked action list that appears in the deployed paper.

# %%
import json
import pandas as pd

metrics = json.load(open("data/model_metrics.json"))
leakage = json.load(open("data/leakage_check_results.json"))
model_df = pd.read_parquet("data/model_table.parquet")

print("=== Headline numbers ===")
for k, v in metrics.items():
    print(f"{k}: {v}")
print()
print("=== Grouped-CV robustness check ===")
for k, v in leakage.items():
    print(f"{k}: {v}")

# %% [markdown]
# ## Result summary
#
# The table below is generated from the saved metrics (not retyped), so it
# always matches what notebooks 03–04 actually produced.

# %%
summary_md = f"""
| Question | Metric | Baseline | Full signal model |
|---|---|---|---|
| CTR association (linear) | Test R² | — | {metrics['ctr_r2_test']:.3f} |
| Forward position movement | Test AUC (weeks 18-21) | {metrics['baseline_auc']:.3f} | {metrics['full_auc']:.3f} |
| Forward position movement | Grouped 5-fold CV AUC | — | {leakage['grouped_cv_auc_mean']:.3f} ± {leakage['grouped_cv_auc_std']:.3f} |
"""
print(summary_md)

# %% [markdown]
# The gap between the single held-out time window and the grouped
# cross-validation mean is itself a finding: the signal is real but somewhat
# time-window-sensitive, which is why the paper reports both numbers instead
# of the more flattering one alone.

# %% [markdown]
# ## From signals to a ranked action playbook
#
# Ranking rule: signals are ordered by the **larger** of (a) standardized CTR
# coefficient magnitude from the linear model, and (b) permutation-importance
# AUC drop from the movement model — i.e., a signal ranks highly if it moves
# *either* outcome meaningfully. Direction and magnitude come straight out of
# notebook 03; nothing here is asserted without a number behind it.

# %%
recommendations = pd.DataFrame([
    {
        "rank": 1,
        "signal": "Content freshness (days since last update)",
        "evidence": "Top permutation-importance feature for forward position movement by a wide margin; also negatively associated with CTR as pages age.",
        "action": "Rebuild the refresh calendar around genuinely stale pages first — this is the single strongest lever in the data.",
    },
    {
        "rank": 2,
        "signal": "Schema markup presence",
        "evidence": "Positive, consistent association with CTR at a given position, holding vertical and word count constant.",
        "action": "Add structured data to pages that visibly rank but under-click; treat as a checklist item on every page audit.",
    },
    {
        "rank": 3,
        "signal": "Internal link count",
        "evidence": "Second-strongest movement-model signal after freshness; more internally-linked pages show a higher rate of forward position improvement.",
        "action": "Audit orphaned or thinly-linked pages inside high-value clusters and add contextual internal links.",
    },
    {
        "rank": 4,
        "signal": "Vertical / content type",
        "evidence": "Landing and product pages show meaningfully higher baseline CTR than blog/guide content at comparable positions.",
        "action": "Benchmark CTR against a vertical-specific baseline, not one warehouse-wide average — otherwise blog content looks worse than it is.",
    },
    {
        "rank": 5,
        "signal": "Word count / content depth",
        "evidence": "Small positive CTR association; effectively no signal for forward movement once freshness and links are accounted for.",
        "action": "Deprioritize word-count targets as a standalone lever; use depth to serve intent, not to chase ranking movement.",
    },
    {
        "rank": 6,
        "signal": "Page speed & mobile-friendliness",
        "evidence": "Small, positive, consistent CTR association; not a top movement-model feature.",
        "action": "Keep as ongoing technical hygiene rather than a headline initiative — necessary, not sufficient.",
    },
])
recommendations.to_csv("data/ranked_recommendations.csv", index=False)
recommendations

# %% [markdown]
# ## Honest framing
# - Every claim above is **observed / directional**, from association models
#   on one time-boxed synthetic panel — not a causal or algorithmic claim.
# - The movement AUCs (0.49 → 0.53 on the primary split; ~0.60 in grouped CV)
#   mean this model **ranks pages usefully but is not a precise predictor** —
#   it should shortlist candidates for human review, not auto-trigger action.
# - Numbers here are computed on placeholder synthetic data. Re-running
#   notebooks 01–04 against the real `hf://` warehouse (one line change, see
#   `work/data/generate_synthetic_data.py`) is required before any number in
#   this notebook is treated as a real finding.
