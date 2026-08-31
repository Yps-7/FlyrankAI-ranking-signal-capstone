# Ranking Signal Analysis — Search Intelligence Capstone

**Lane:** Ranking Signal Analysis
**Deployed paper:** see `submission/paper_url.txt` once published (instructions below)

## What's here

```
work/
  data/
    generate_synthetic_data.py       # synthetic placeholder generator + real-dataset swap instructions
    signals_panel.csv                # generated placeholder panel (900 pages x 26 weeks)
    model_table.parquet              # engineered features + labels (output of 02)
    model_metrics.json               # saved metrics (output of 03, read by the paper + capstone notebook)
    leakage_check_results.json       # saved grouped-CV results (output of 04)
    ranked_recommendations.csv       # final playbook (output of capstone notebook)
  01_data_intake_eda.ipynb           # DuckDB load, schema/quality checks, EDA
  02_feature_engineering.ipynb       # leakage-safe forward label + feature build
  03_baseline_and_signal_model.ipynb # baseline vs. full model, CTR coefficients, feature importance
  04_validation_leakage_checks.ipynb # grouped cross-validation + leakage audit
  capstone_ranking_signal_analysis.ipynb  # synthesis notebook: final numbers + ranked recommendations
paper/
  index.html                         # the deployed research paper (single self-contained file)
  assets/                            # chart PNGs referenced by index.html
submission/
  paper_url.txt                      # MUST contain exactly one line: the live URL of paper/index.html
```

Every `.ipynb` above is already executed — outputs, tables, and charts are baked in. Open them directly to review, or re-run top to bottom to regenerate everything (see below).

## This currently runs on synthetic data because i can't access the warehouse data from HF.

`work/data/generate_synthetic_data.py` builds a **placeholder panel schema-matched to the real FlyRank ML Internship warehouse** — same columns, same shape, plausible (but fabricated) relationships. It exists so the whole pipeline runs today without a Hugging Face token. All numbers in the notebooks and on the deployed paper currently reflect that placeholder, and the paper says so explicitly (see its "Data status" banner).

**To finalize with real data:**

1. Get your Hugging Face read token for the FlyRank warehouse.
2. In `work/01_data_intake_eda.ipynb`, replace the load cell:
   ```python
   con = duckdb.connect()
   con.sql("SET hf_token='<your_token>';")
   panel = con.sql(
       "SELECT * FROM read_parquet('hf://datasets/<flyrank-repo>/*.parquet')"
   ).df()
   ```
   Every other cell in every notebook is unchanged — column names in the real warehouse should match `generate_synthetic_data.py`'s schema; adjust the feature list in notebooks 02/03/04 if any real column is named differently.
3. Re-run in order: `01 → 02 → 03 → 04 → capstone`. Metrics, charts, and `ranked_recommendations.csv` all regenerate automatically.
4. Re-run the chart-export cells so `paper/assets/*.png` reflect the real numbers, then update the numeric callouts in `paper/index.html` (stat row, results table, results-section prose) to match.
5. Redeploy the paper (below) and confirm `submission/paper_url.txt` still points at the live page.
