# -*- coding: utf-8 -*-
"""
eeg_prepost_stats.py
======================

Pre-post statistical comparison for the EEG feature table produced by
aggregate_features.py (wide_feature_table.csv: one row per subject,
columns "{task}_{ROI}_{feature}_pre" / "..._post").

Same statistical spirit as the cognitive-test pre-post analysis
(Shapiro-Wilk -> paired t-test/Wilcoxon -> effect size -> FDR), adapted
for EEG's much larger number of features per task (~35: 5 ROI x 4 bands
relative power, + 5 ROI x 2 ratios, + 5 ROI x entropy):

  - FDR correction is applied SEPARATELY PER TASK (its own family of
    ~35 tests), not across all 175 features at once -- matches the
    earlier project decision to keep Memory's two phases (3A encoding,
    3B retrieval) as fully independent families, and avoids
    over-correcting power for tasks that are conceptually distinct
    questions (Attention, Language, Math are separate hypotheses too).

  - Given 175 total comparisons, individual estimation plots per feature
    (as used for the 5 cognitive tests) would be impractical to review.
    Instead, this produces a heatmap per task (ROI x measure grid,
    colored by effect size, starred by FDR-significance) -- much more
    reviewable at this scale.

Usage
-----
    python eeg_prepost_stats.py wide_feature_table.csv
"""

import sys
import re
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

ALPHA = 0.05
N_BOOT = 5000
RNG = np.random.default_rng(42)

TASKS = ["3A", "3B", "4", "5", "6"]
ROIS = ["Frontal", "Central", "Parietal", "Temporal", "Occipital"]
BANDS = ["delta", "theta", "alpha", "beta"]
MEASURES = (
    [f"{b}_relpower" for b in BANDS]
    + ["theta_alpha_ratio", "theta_beta_ratio"]
    + ["entropy"]
)


# ----------------------------------------------------------------------
# Core paired-test logic (same as the cognitive-test script)
# ----------------------------------------------------------------------
def cohens_dz(diff):
    return diff.mean() / diff.std(ddof=1)


def rank_biserial_wilcoxon(diff):
    diff = diff[diff != 0]
    if len(diff) == 0:
        return np.nan
    ranks = stats.rankdata(np.abs(diff))
    w_pos = ranks[diff > 0].sum()
    w_neg = ranks[diff < 0].sum()
    return (w_pos - w_neg) / (w_pos + w_neg)


def bootstrap_ci_mean(x, n_boot=N_BOOT, alpha=ALPHA):
    x = np.asarray(x)
    n = len(x)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        boot_means[i] = RNG.choice(x, size=n, replace=True).mean()
    lo = np.percentile(boot_means, 100 * (alpha / 2))
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return lo, hi


def fdr_bh(pvals):
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    valid = ~np.isnan(pvals)
    out = np.full(n, np.nan)
    if valid.sum() == 0:
        return out
    p_valid = pvals[valid]
    order = np.argsort(p_valid)
    ranked = p_valid[order]
    n_valid = len(p_valid)
    adj = ranked * n_valid / (np.arange(n_valid) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    result_valid = np.empty(n_valid)
    result_valid[order] = adj
    out[valid] = result_valid
    return out


def sig_stars(p):
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return "ns"


def analyze_one_feature(df, pre_col, post_col):
    sub = df[[pre_col, post_col]].dropna()
    n = len(sub)
    if n < 3:
        return None

    pre = sub[pre_col].to_numpy()
    post = sub[post_col].to_numpy()
    diff = post - pre

    sw_p = stats.shapiro(diff)[1] if n >= 3 else np.nan
    normal = sw_p >= ALPHA

    if normal:
        test_used = "Paired t-test"
        stat_value, p_val = stats.ttest_rel(post, pre)
        effect = cohens_dz(diff)
        effect_label = "Cohen's dz"
    else:
        test_used = "Wilcoxon signed-rank"
        try:
            stat_value, p_val = stats.wilcoxon(post, pre, zero_method="wilcox")
        except ValueError:
            return None  # e.g. all differences are zero
        effect = rank_biserial_wilcoxon(diff)
        effect_label = "Rank-biserial r"

    ci_lo, ci_hi = bootstrap_ci_mean(diff)

    return {
        "N": n,
        "Mean_Pre": round(pre.mean(), 4),
        "Mean_Post": round(post.mean(), 4),
        "Mean_Diff": round(diff.mean(), 4),
        "Boot_95CI_Low": round(ci_lo, 4),
        "Boot_95CI_High": round(ci_hi, 4),
        "Shapiro_p": round(sw_p, 4),
        "Test_Used": test_used,
        "Statistic": round(stat_value, 4),
        "p_value": p_val,
        "Effect_Size_Type": effect_label,
        "Effect_Size": round(effect, 4),
    }


# ----------------------------------------------------------------------
# Run analysis for every task x ROI x measure combination
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Feature families for FDR correction. Each task gets FOUR SEPARATE
# families (not one 35-test family) -- matches the earlier project
# decision: band power has a "band" dimension (5 ROI x 4 bands = 20
# tests) so its family is naturally larger, while ratios and entropy
# have no band dimension (5 ROI each = 5 tests) so each gets its own
# smaller family. Forcing everything into one family would over-correct
# the smaller feature types relative to what was agreed.
# ----------------------------------------------------------------------
FEATURE_FAMILIES = {
    "band_power": [f"{b}_relpower" for b in BANDS],       # 4 measures x 5 ROI = 20
    "theta_alpha_ratio": ["theta_alpha_ratio"],             # 1 measure x 5 ROI = 5
    "theta_beta_ratio": ["theta_beta_ratio"],                # 1 measure x 5 ROI = 5
    "entropy": ["entropy"],                                   # 1 measure x 5 ROI = 5
}


def run_all_tasks(df):
    all_results = {}  # task -> DataFrame

    for task in TASKS:
        task_rows = []
        for family_name, family_measures in FEATURE_FAMILIES.items():
            family_rows = []
            for roi in ROIS:
                for measure in family_measures:
                    base = f"{task}_{roi}_{measure}"
                    pre_col = f"{base}_pre"
                    post_col = f"{base}_post"
                    if pre_col not in df.columns or post_col not in df.columns:
                        continue
                    result = analyze_one_feature(df, pre_col, post_col)
                    if result is None:
                        continue
                    result["Task"] = task
                    result["ROI"] = roi
                    result["Measure"] = measure
                    result["Family"] = family_name
                    family_rows.append(result)

            if not family_rows:
                continue

            # FDR applied WITHIN this family only (e.g. within the 20
            # band-power tests, separately from the 5 entropy tests)
            family_df = pd.DataFrame(family_rows)
            family_df["p_adj_FDR"] = fdr_bh(family_df["p_value"].to_numpy())
            task_rows.append(family_df)

        if not task_rows:
            print(f"  WARNING: no valid features found for task {task}")
            continue

        task_df = pd.concat(task_rows, ignore_index=True)
        task_df["Significant_after_FDR"] = task_df["p_adj_FDR"] < ALPHA
        task_df["p_value"] = task_df["p_value"].round(4)
        task_df["p_adj_FDR"] = task_df["p_adj_FDR"].round(4)

        cols_order = ["Task", "Family", "ROI", "Measure", "N", "Mean_Pre", "Mean_Post", "Mean_Diff",
                      "Boot_95CI_Low", "Boot_95CI_High", "Shapiro_p", "Test_Used",
                      "Statistic", "p_value", "p_adj_FDR", "Significant_after_FDR",
                      "Effect_Size_Type", "Effect_Size"]
        task_df = task_df[cols_order]
        all_results[task] = task_df

    return all_results


# ----------------------------------------------------------------------
# Heatmap visualization: ROI x Measure grid, colored by effect size,
# starred by FDR significance -- one per task
# ----------------------------------------------------------------------
def make_heatmap(task_df, task_label, out_path):
    pivot_effect = task_df.pivot(index="ROI", columns="Measure", values="Effect_Size")
    pivot_effect = pivot_effect.reindex(index=ROIS, columns=MEASURES)

    pivot_stars = task_df.pivot(index="ROI", columns="Measure", values="p_adj_FDR")
    pivot_stars = pivot_stars.reindex(index=ROIS, columns=MEASURES)

    fig, ax = plt.subplots(figsize=(10, 5))
    vmax = np.nanmax(np.abs(pivot_effect.to_numpy())) if not np.all(np.isnan(pivot_effect.to_numpy())) else 1
    im = ax.imshow(pivot_effect.to_numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(MEASURES)))
    ax.set_xticklabels(MEASURES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(ROIS)))
    ax.set_yticklabels(ROIS, fontsize=9)

    for i in range(len(ROIS)):
        for j in range(len(MEASURES)):
            p = pivot_stars.to_numpy()[i, j]
            stars = sig_stars(p) if not np.isnan(p) else ""
            if stars and stars != "ns":
                ax.text(j, i, stars, ha="center", va="center", fontsize=11,
                         fontweight="bold", color="black")

    ax.set_title(f"Task {task_label} -- Effect size (color) x FDR significance (stars)", fontsize=11)
    fig.colorbar(im, ax=ax, label="Effect size (Cohen's dz / rank-biserial r)", shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(csv_path):
    df = pd.read_csv(csv_path)
    print(f"Loaded {csv_path}: {df.shape[0]} subjects, {df.shape[1]} columns")

    all_results = run_all_tasks(df)

    combined = pd.concat(all_results.values(), ignore_index=True)
    combined.to_csv("eeg_prepost_stats_summary.csv", index=False)
    print(f"\nSaved combined summary: eeg_prepost_stats_summary.csv ({len(combined)} rows)")

    n_sig = combined["Significant_after_FDR"].sum()
    print(f"Significant after FDR (per-task correction): {n_sig}/{len(combined)}")
    if n_sig > 0:
        print("\nSignificant results:")
        sig_df = combined[combined["Significant_after_FDR"]]
        print(sig_df[["Task", "ROI", "Measure", "Mean_Diff", "p_adj_FDR", "Effect_Size"]].to_string(index=False))

    for task, task_df in all_results.items():
        task_csv = f"eeg_prepost_stats_task_{task}.csv"
        task_df.to_csv(task_csv, index=False)
        heatmap_path = f"eeg_prepost_heatmap_task_{task}.png"
        make_heatmap(task_df, task, heatmap_path)
        print(f"Saved: {task_csv}, {heatmap_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python eeg_prepost_stats.py wide_feature_table.csv")
        sys.exit(1)
    main(sys.argv[1])
