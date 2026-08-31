"""
Synthetic FlyRank-style Search Intelligence warehouse — PLACEHOLDER DATA ONLY.

This mimics the *shape* of the real FlyRank ML Internship dataset (page-level
weekly search performance + content/technical signals) so every notebook in
this repo runs end-to-end right now. It contains no real domains, URLs,
client names, or queries — only anonymized page IDs.

>>> SWAP-IN INSTRUCTIONS FOR THE REAL DATASET <<<
When you have your Hugging Face read token:
    import duckdb
    con = duckdb.connect()
    con.sql("SET hf_token='<your_token>';")
    df = con.sql("SELECT * FROM read_parquet('hf://datasets/<flyrank-repo>/*.parquet')").df()
Replace the `load_panel()` call in 01_data_intake_eda.ipynb with that query,
keep every downstream feature/label/model cell identical, and re-run.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

N_PAGES = 900
N_WEEKS = 26
VERTICALS = ["blog", "product", "guide", "landing", "news"]
VERTICAL_BASE_POSITION = {"blog": 18, "product": 14, "guide": 22, "landing": 11, "news": 25}


def generate_panel() -> pd.DataFrame:
    page_ids = [f"page_{i:05d}" for i in range(N_PAGES)]
    verticals = RNG.choice(VERTICALS, size=N_PAGES, p=[0.32, 0.22, 0.24, 0.12, 0.10])

    # static-ish content/technical signals per page (can drift slightly week to week)
    word_count = np.clip(RNG.normal(1200, 500, N_PAGES), 150, 4500)
    has_schema = RNG.choice([0, 1], size=N_PAGES, p=[0.55, 0.45])
    internal_links = np.clip(RNG.poisson(9, N_PAGES), 0, 60)
    external_links = np.clip(RNG.poisson(3, N_PAGES), 0, 25)
    mobile_friendly = np.clip(RNG.normal(78, 14, N_PAGES), 20, 100)
    page_speed = np.clip(RNG.normal(70, 16, N_PAGES), 15, 100)
    title_length = np.clip(RNG.normal(56, 14, N_PAGES), 15, 100)
    meta_desc_length = np.clip(RNG.normal(135, 35, N_PAGES), 20, 320)
    h1_count = RNG.choice([0, 1, 1, 1, 2], size=N_PAGES)
    image_count = np.clip(RNG.poisson(6, N_PAGES), 0, 40)
    alt_text_coverage = np.clip(RNG.beta(3, 1.5, N_PAGES), 0, 1)
    readability = np.clip(RNG.normal(60, 15, N_PAGES), 10, 100)
    initial_update_age = RNG.integers(1, 900, N_PAGES)  # days since last content update, at week 0

    base_position = np.array([VERTICAL_BASE_POSITION[v] for v in verticals])
    # true (hidden) association strengths used to *simulate* the world —
    # the notebooks re-discover these from data, they never read this dict.
    true_position_effect = (
        base_position
        - 0.0022 * (word_count - 1200)
        - 2.1 * has_schema
        - 0.05 * internal_links
        - 0.045 * (page_speed - 70)
        - 0.02 * (mobile_friendly - 78)
        + 0.0025 * np.abs(title_length - 58)
        - 0.01 * (readability - 60)
        + RNG.normal(0, 2.2, N_PAGES)
    )
    true_position_effect = np.clip(true_position_effect, 1, 95)

    rows = []
    update_age = initial_update_age.astype(float).copy()
    cur_position = true_position_effect.copy()

    for week in range(N_WEEKS):
        # freshness decays each week unless (simulated) refreshed
        refreshed_this_week = RNG.random(N_PAGES) < 0.03
        update_age = np.where(refreshed_this_week, RNG.integers(0, 5, N_PAGES), update_age + 7)

        # freshness gives a small, decaying pull toward better position after a refresh
        freshness_pull = np.where(update_age < 21, -1.6, 0.0)
        seasonal = 1.5 * np.sin(2 * np.pi * week / 12)
        noise = RNG.normal(0, 1.6, N_PAGES)

        cur_position = np.clip(
            0.92 * cur_position + 0.08 * true_position_effect + freshness_pull + seasonal * 0.15 + noise,
            1, 100,
        )

        # impressions driven by vertical + mild randomness; clicks driven by position + CTR-affecting signals
        impressions = np.clip(
            RNG.negative_binomial(6, 0.35, N_PAGES) * (1 + (verticals == "landing") * 0.6),
            5, None,
        ).astype(int)

        expected_ctr = np.clip(
            0.28 * np.exp(-cur_position / 9)
            + 0.02 * has_schema
            + 0.00015 * (mobile_friendly - 78)
            + RNG.normal(0, 0.01, N_PAGES),
            0.002, 0.55,
        )
        clicks = RNG.binomial(impressions, expected_ctr)
        ctr = np.where(impressions > 0, clicks / impressions, 0.0)

        rows.append(
            pd.DataFrame(
                {
                    "page_id": page_ids,
                    "vertical": verticals,
                    "week": week,
                    "impressions": impressions,
                    "clicks": clicks,
                    "ctr": ctr,
                    "avg_position": cur_position,
                    "word_count": word_count.round().astype(int),
                    "has_schema_markup": has_schema,
                    "num_internal_links": internal_links,
                    "num_external_links": external_links,
                    "days_since_last_update": update_age.round().astype(int),
                    "mobile_friendly_score": mobile_friendly.round(1),
                    "page_speed_score": page_speed.round(1),
                    "title_length": title_length.round().astype(int),
                    "meta_description_length": meta_desc_length.round().astype(int),
                    "h1_count": h1_count,
                    "image_count": image_count,
                    "alt_text_coverage": alt_text_coverage.round(3),
                    "readability_score": readability.round(1),
                }
            )
        )

    panel = pd.concat(rows, ignore_index=True)
    return panel


if __name__ == "__main__":
    df = generate_panel()
    out_path = "/home/claude/flyrank-capstone/work/data/signals_panel.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df):,} rows -> {out_path}")
    print(df.head())
