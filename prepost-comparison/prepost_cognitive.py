"""
Pre-Post Cognitive Test Analysis (Pooled N, BrainTrain intervention)
=====================================================================

Purpose
-------
Runs paired pre-post analysis for 5 cognitive tests (MMSE, TMT-A, TMT-B,
Digit Span Forward, Digit Span Backward) across the full sample (N>40),
without splitting by usage group. This is the PRIMARY analysis; the
Low/High usage subgroup analysis should be run separately afterward.

Input
-----
A CSV file in WIDE format, one row per subject, with columns:
    subject_id, MMSE_pre, MMSE_post, TMTA_pre, TMTA_post,
    TMTB_pre, TMTB_post, DSF_pre, DSF_post, DSB_pre, DSB_post
(demographic columns like age/sex/education are fine to keep, they are
simply ignored by this script)

Missing values are handled pairwise per test (a subject missing MMSE_post
is simply dropped from the MMSE analysis but kept for other tests).

Method logic
------------
For each test:
  1. Compute pre->post difference (direction-corrected: for TMT-A/TMT-B,
     "improvement" = decrease in time, so we flip sign internally when
     reporting effect size/direction, but statistics are run on raw
     signed differences as given).
  2. Shapiro-Wilk test on the difference scores to check normality.
  3. If normal (p >= .05): paired t-test + Cohen's dz (mean diff / sd diff).
     If non-normal (p < .05): Wilcoxon signed-rank test + matched-pairs
     rank-biserial correlation as effect size.
  4. Bonferroni-Holm / Benjamini-Hochberg FDR correction is applied across
     the 5 tests (family-wise, since they are analyzed together as one
     cognitive battery).
  5. Bootstrap 95% CI on the mean difference (percentile method, 5000
     resamples) is reported alongside parametric/nonparametric CI.

Output
------
- Prints a formatted summary table to console
- Saves summary_results.csv
- Saves a before-after paired plot per test (paired_plot_<test>.png)

Usage
-----
    python prepost_cognitive_analysis.py your_data.csv
"""

import sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Config: define each test, its pre/post column names, and whether a
# LOWER post-score indicates improvement (True) or a HIGHER score does
# (False). This matters only for the "direction of improvement" label,
# not for the raw statistical test itself.
# ----------------------------------------------------------------------
TESTS = {
    "MMSE": {"pre": "MMSE_pre", "post": "MMSE_post", "lower_is_better": False},
    "TMT-A": {"pre": "TMTA_pre", "post": "TMTA_post", "lower_is_better": True},
    "TMT-B": {"pre": "TMTB_pre", "post": "TMTB_post", "lower_is_better": True},
    "Digit Span Forward": {"pre": "DSF_pre", "post": "DSF_post", "lower_is_better": False},
    "Digit Span Backward": {"pre": "DSB_pre", "post": "DSB_post", "lower_is_better": False},
}

N_BOOT = 5000
ALPHA = 0.05
RNG = np.random.default_rng(42)


def bootstrap_ci_mean(x, n_boot=N_BOOT, alpha=ALPHA):
    """Percentile bootstrap CI for the mean of x."""
    x = np.asarray(x)
    boot_means = np.empty(n_boot)
    n = len(x)
    for i in range(n_boot):
        sample = RNG.choice(x, size=n, replace=True)
        boot_means[i] = sample.mean()
    lo = np.percentile(boot_means, 100 * (alpha / 2))
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return lo, hi


def cohens_dz(diff):
    """Effect size for paired t-test: mean diff / sd of diff."""
    return diff.mean() / diff.std(ddof=1)


def rank_biserial_wilcoxon(diff):
    """
    Matched-pairs rank-biserial correlation as effect size for Wilcoxon
    signed-rank test. r = (W+ - W-) / (W+ + W-)  where W+/W- are sums of
    positive/negative ranks.
    """
    diff = diff[diff != 0]  # ties with zero are dropped, standard Wilcoxon convention
    ranks = stats.rankdata(np.abs(diff))
    w_pos = ranks[diff > 0].sum()
    w_neg = ranks[diff < 0].sum()
    return (w_pos - w_neg) / (w_pos + w_neg)


def analyze_one_test(df, test_name, pre_col, post_col, lower_is_better):
    sub = df[[pre_col, post_col]].dropna()
    n = len(sub)
    pre = sub[pre_col].to_numpy()
    post = sub[post_col].to_numpy()
    diff = post - pre  # raw signed difference, post minus pre

    mean_pre, sd_pre = pre.mean(), pre.std(ddof=1)
    mean_post, sd_post = post.mean(), post.std(ddof=1)
    mean_diff = diff.mean()

    # normality check on the difference scores
    if n >= 3:
        sw_stat, sw_p = stats.shapiro(diff)
    else:
        sw_p = np.nan

    if n < 3:
        raise ValueError(f"{test_name}: fewer than 3 paired observations (n={n}); cannot test.")

    normal = sw_p >= ALPHA

    if normal:
        test_used = "Paired t-test"
        t_stat, p_val = stats.ttest_rel(post, pre)
        effect_size = cohens_dz(diff)
        effect_label = "Cohen's dz"
        stat_value = t_stat
    else:
        test_used = "Wilcoxon signed-rank"
        # zero_method='wilcox' drops zero-differences, standard default
        w_stat, p_val = stats.wilcoxon(post, pre, zero_method="wilcox")
        effect_size = rank_biserial_wilcoxon(diff)
        effect_label = "Rank-biserial r"
        stat_value = w_stat

    ci_lo, ci_hi = bootstrap_ci_mean(diff)

    # human-readable improvement direction
    if lower_is_better:
        improved = mean_diff < 0
    else:
        improved = mean_diff > 0
    direction = "Improved" if improved else ("No change" if mean_diff == 0 else "Worsened")

    return {
        "Test": test_name,
        "N": n,
        "Mean_Pre": round(mean_pre, 2),
        "SD_Pre": round(sd_pre, 2),
        "Mean_Post": round(mean_post, 2),
        "SD_Post": round(sd_post, 2),
        "Mean_Diff": round(mean_diff, 2),
        "Boot_95CI_Low": round(ci_lo, 2),
        "Boot_95CI_High": round(ci_hi, 2),
        "Shapiro_p": round(sw_p, 4),
        "Test_Used": test_used,
        "Statistic": round(stat_value, 3),
        "p_value": p_val,
        "Effect_Size_Type": effect_label,
        "Effect_Size": round(effect_size, 3),
        "Direction": direction,
        "_diff_raw": diff,  # kept for plotting, stripped before CSV export
        "_pre_raw": pre,
        "_post_raw": post,
    }


def fdr_bh(pvals):
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values."""
    pvals = np.asarray(pvals)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    adj = ranked * n / (np.arange(n) + 1)
    # enforce monotonicity
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty(n)
    out[order] = adj
    return out


def sig_stars(p):
    """Standard significance star convention, based on the ADJUSTED p-value."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"


def _draw_paired_panel(ax, result, panel_label=None, title_fontsize=11):
    """
    Draws one pre->post paired panel (spaghetti plot + significance bracket)
    onto a given matplotlib Axes. Shared by both the single-figure export
    (make_paired_plot) and the combined 5-panel manuscript figure
    (make_combined_figure), so the two stay visually identical.
    """
    pre = result["_pre_raw"]
    post = result["_post_raw"]
    n = len(pre)
    p_adj = result["p_adj_FDR"]
    stars = sig_stars(p_adj)

    for i in range(n):
        ax.plot([0, 1], [pre[i], post[i]], color="gray", alpha=0.4, linewidth=0.8)
    ax.scatter(np.zeros(n), pre, color="#4C72B0", zorder=3, s=22, label="Pre")
    ax.scatter(np.ones(n), post, color="#DD8452", zorder=3, s=22, label="Post")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pre", "Post"])
    ax.set_xlim(-0.3, 1.3)

    y_max = max(pre.max(), post.max())
    y_min = min(pre.min(), post.min())
    y_range = y_max - y_min if y_max > y_min else 1.0
    bar_y = y_max + 0.08 * y_range
    tick_h = 0.02 * y_range

    ax.plot([0, 0, 1, 1], [bar_y - tick_h, bar_y, bar_y, bar_y - tick_h],
            color="black", linewidth=1.0)
    ax.text(0.5, bar_y + 0.01 * y_range, stars,
            ha="center", va="bottom",
            fontsize=13 if stars != "ns" else 9,
            fontweight="bold" if stars != "ns" else "normal")

    ax.set_ylim(y_min - 0.05 * y_range, bar_y + 0.15 * y_range)

    title = f"{result['Test']} (n={n})\n{result['Test_Used']}, p_FDR = {p_adj:.4f}"
    if panel_label:
        # panel_label (A, B, C...) placed top-left, outside the axes, manuscript style
        ax.text(-0.15, 1.12, panel_label, transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="top", ha="left")
    ax.set_title(title, fontsize=title_fontsize)


def make_paired_plot(result, out_path):
    fig, ax = plt.subplots(figsize=(4, 5))
    _draw_paired_panel(ax, result)
    ax.set_ylabel("Score")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_combined_figure(results, out_path):
    """
    Combines all test panels into one figure (2x3 grid), for manuscript
    submission (e.g. Figure 2: pre-post change across the cognitive
    battery). Panels are labeled A, B, C... automatically.
    """
    n_tests = len(results)
    n_cols = 3
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 5 * n_rows))
    axes = axes.flatten()
    panel_labels = [chr(ord("A") + i) for i in range(n_tests)]

    for ax, result, label in zip(axes, results, panel_labels):
        _draw_paired_panel(ax, result, panel_label=label, title_fontsize=10)

    # hide any unused subplot slots (5 tests into a 2x3 grid leaves 1 empty)
    for ax in axes[n_tests:]:
        ax.axis("off")

    axes[0].set_ylabel("Score")
    axes[n_cols].set_ylabel("Score")  # first axis of the second row

    # single shared legend for the whole figure, placed below all panels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.02), frameon=False)

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(csv_path):
    df = pd.read_csv(csv_path)
    results = []
    for test_name, cfg in TESTS.items():
        if cfg["pre"] not in df.columns or cfg["post"] not in df.columns:
            print(f"WARNING: columns for {test_name} not found, skipping.")
            continue
        res = analyze_one_test(df, test_name, cfg["pre"], cfg["post"], cfg["lower_is_better"])
        results.append(res)

    pvals = [r["p_value"] for r in results]
    p_adj = fdr_bh(pvals)
    for r, pa in zip(results, p_adj):
        r["p_adj_FDR"] = round(pa, 4)
        r["Significant_after_FDR"] = pa < ALPHA

    # build export table (drop raw arrays)
    export_rows = []
    for r in results:
        row = {k: v for k, v in r.items() if not k.startswith("_")}
        export_rows.append(row)
    out_df = pd.DataFrame(export_rows)
    out_df["p_value"] = out_df["p_value"].round(4)

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    print("\n=== Pre-Post Summary (Pooled N) ===\n")
    print(out_df.to_string(index=False))

    out_csv = "summary_results.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    for r in results:
        fname = f"paired_plot_{r['Test'].replace(' ', '_').replace('-', '')}.png"
        make_paired_plot(r, fname)
        print(f"Saved: {fname}")

    combined_fname = "combined_panel_all_tests.png"
    make_combined_figure(results, combined_fname)
    print(f"Saved: {combined_fname}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python prepost_cognitive_analysis.py your_data.csv")
        sys.exit(1)
    main(sys.argv[1])