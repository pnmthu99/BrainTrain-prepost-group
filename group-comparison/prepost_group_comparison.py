"""
Pre-Post Cognitive Analysis: Low vs High Usage Group Comparison
==================================================================

Purpose
-------
Tests whether the pre->post change on 5 cognitive tests (MMSE, TMT-A,
TMT-B, Digit Span Forward, Digit Span Backward) differs between the Low
usage group and the High usage group (>30 min/day threshold).

Design logic
------------
PRIMARY analysis = Group x Time interaction.
With only 2 timepoints, this interaction is mathematically equivalent to
comparing the pre->post difference scores (delta = post - pre) between
the two groups with an independent-samples test. This is simpler and
more transparent than fitting a full mixed model with only 2 waves, and
avoids convergence issues with small per-group N (e.g. 17 vs 19).

SECONDARY analysis = simple effects, i.e. the within-group paired
pre-post test for Low and for High separately (same logic as the pooled
single-group script). These help interpret the direction/source of a
significant interaction, but are not the primary hypothesis test.

Input
-----
Wide-format CSV with columns:
    subject_id, Group (values: "Low" or "High"),
    MMSE_pre, MMSE_post, TMTA_pre, TMTA_post, TMTB_pre, TMTB_post,
    DSF_pre, DSF_post, DSB_pre, DSB_post

Output
------
- interaction_summary.csv       (primary: Group x Time test per cognitive test)
- simple_effects_summary.csv    (secondary: within-group paired test per group per test)
- interaction_plot_<test>.png   (mean +/- SEM trajectory, Low vs High, per test)
- combined_interaction_panel.png (all 5 tests side by side, manuscript-ready)

Usage
-----
    python prepost_group_comparison.py your_data.csv
"""

import sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import pingouin as pg

TESTS = {
    "MMSE": {"pre": "MMSE_pre", "post": "MMSE_post", "lower_is_better": False},
    "TMT-A": {"pre": "TMTA_pre", "post": "TMTA_post", "lower_is_better": True},
    "TMT-B": {"pre": "TMTB_pre", "post": "TMTB_post", "lower_is_better": True},
    "Digit Span Forward": {"pre": "DSF_pre", "post": "DSF_post", "lower_is_better": False},
    "Digit Span Backward": {"pre": "DSB_pre", "post": "DSB_post", "lower_is_better": False},
}

ALPHA = 0.05
GROUP_ORDER = ["Low", "High"]
GROUP_COLORS = {"Low": "#4C72B0", "High": "#DD8452"}


# ----------------------------------------------------------------------
# Effect sizes / helpers
# ----------------------------------------------------------------------
def cohens_dz(diff):
    return diff.mean() / diff.std(ddof=1)


def hedges_g(x1, x2):
    """Bias-corrected Cohen's d for two independent samples (Hedges' g)."""
    n1, n2 = len(x1), len(x2)
    s1, s2 = x1.std(ddof=1), x2.std(ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    d = (x1.mean() - x2.mean()) / pooled_sd
    correction = 1 - (3 / (4 * (n1 + n2) - 9))
    return d * correction


def rank_biserial_mw(x1, x2, u_stat):
    """Rank-biserial correlation effect size for Mann-Whitney U."""
    n1, n2 = len(x1), len(x2)
    return 1 - (2 * u_stat) / (n1 * n2)


def rank_biserial_wilcoxon(diff):
    diff = diff[diff != 0]
    ranks = stats.rankdata(np.abs(diff))
    w_pos = ranks[diff > 0].sum()
    w_neg = ranks[diff < 0].sum()
    return (w_pos - w_neg) / (w_pos + w_neg)


def fdr_bh(pvals):
    pvals = np.asarray(pvals)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    adj = ranked * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty(n)
    out[order] = adj
    return out


def sig_stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"


# ----------------------------------------------------------------------
# PRIMARY: Group x Time interaction (= between-group test on delta)
# ----------------------------------------------------------------------
def analyze_interaction(df, test_name, pre_col, post_col, lower_is_better):
    sub = df[["Group", pre_col, post_col]].dropna()
    low = sub[sub["Group"] == "Low"]
    high = sub[sub["Group"] == "High"]

    delta_low = (low[post_col] - low[pre_col]).to_numpy()
    delta_high = (high[post_col] - high[pre_col]).to_numpy()
    n_low, n_high = len(delta_low), len(delta_high)

    if n_low < 3 or n_high < 3:
        raise ValueError(f"{test_name}: insufficient N in one group (Low={n_low}, High={n_high}).")

    sw_low_p = stats.shapiro(delta_low)[1]
    sw_high_p = stats.shapiro(delta_high)[1]
    both_normal = (sw_low_p >= ALPHA) and (sw_high_p >= ALPHA)

    if both_normal:
        test_used = "Welch's t-test"
        t_stat, p_val = stats.ttest_ind(delta_high, delta_low, equal_var=False)
        eff = hedges_g(delta_high, delta_low)
        eff_label = "Hedges' g"
        stat_value = t_stat
    else:
        test_used = "Mann-Whitney U"
        u_stat, p_val = stats.mannwhitneyu(delta_high, delta_low, alternative="two-sided")
        eff = rank_biserial_mw(delta_high, delta_low, u_stat)
        eff_label = "Rank-biserial r"
        stat_value = u_stat

    return {
        "Test": test_name,
        "N_Low": n_low,
        "N_High": n_high,
        "Low_Mean_Pre": round(low[pre_col].mean(), 2),
        "Low_Mean_Post": round(low[post_col].mean(), 2),
        "Low_Mean_Diff": round(delta_low.mean(), 2),
        "Low_SEM_Diff": round(delta_low.std(ddof=1) / np.sqrt(n_low), 2),
        "High_Mean_Pre": round(high[pre_col].mean(), 2),
        "High_Mean_Post": round(high[post_col].mean(), 2),
        "High_Mean_Diff": round(delta_high.mean(), 2),
        "High_SEM_Diff": round(delta_high.std(ddof=1) / np.sqrt(n_high), 2),
        "Shapiro_Low_p": round(sw_low_p, 4),
        "Shapiro_High_p": round(sw_high_p, 4),
        "Interaction_Test": test_used,
        "Statistic": round(stat_value, 3),
        "p_value": p_val,
        "Effect_Size_Type": eff_label,
        "Effect_Size": round(eff, 3),
        "lower_is_better": lower_is_better,
        "_low_raw": low[[pre_col, post_col]].to_numpy(),
        "_high_raw": high[[pre_col, post_col]].to_numpy(),
    }


# ----------------------------------------------------------------------
# SUPPLEMENTARY: between-group comparison AT POST ONLY (Low-1m vs High-1m)
# This is the "top bracket" comparison seen in Prism-style plots: it asks
# whether the two groups differ at the post timepoint specifically, not
# whether they differ in AMOUNT of change (that's analyze_interaction).
# A baseline (0m) between-group comparison is also returned, useful to
# confirm the two groups didn't already differ before the intervention.
# ----------------------------------------------------------------------
def analyze_between_group_at_timepoint(df, test_name, col, timepoint_label):
    sub = df[["Group", col]].dropna()
    low = sub.loc[sub["Group"] == "Low", col].to_numpy()
    high = sub.loc[sub["Group"] == "High", col].to_numpy()
    n_low, n_high = len(low), len(high)

    sw_low_p = stats.shapiro(low)[1]
    sw_high_p = stats.shapiro(high)[1]
    both_normal = (sw_low_p >= ALPHA) and (sw_high_p >= ALPHA)

    if both_normal:
        test_used = "Welch's t-test"
        stat_value, p_val = stats.ttest_ind(high, low, equal_var=False)
        eff = hedges_g(high, low)
        eff_label = "Hedges' g"
    else:
        test_used = "Mann-Whitney U"
        stat_value, p_val = stats.mannwhitneyu(high, low, alternative="two-sided")
        eff = rank_biserial_mw(high, low, stat_value)
        eff_label = "Rank-biserial r"

    return {
        "Test": test_name,
        "Timepoint": timepoint_label,
        "N_Low": n_low,
        "N_High": n_high,
        "Low_Median": round(np.median(low), 2),
        "High_Median": round(np.median(high), 2),
        "Test_Used": test_used,
        "Statistic": round(stat_value, 3),
        "p_value": p_val,
        "Effect_Size_Type": eff_label,
        "Effect_Size": round(eff, 3),
    }



# ----------------------------------------------------------------------
# SECONDARY: within-group paired pre-post test (simple effect)
# ----------------------------------------------------------------------
def analyze_simple_effect(df_group, group_label, test_name, pre_col, post_col, lower_is_better):
    sub = df_group[[pre_col, post_col]].dropna()
    n = len(sub)
    pre = sub[pre_col].to_numpy()
    post = sub[post_col].to_numpy()
    diff = post - pre

    if n < 3:
        raise ValueError(f"{test_name} ({group_label}): fewer than 3 paired obs (n={n}).")

    sw_p = stats.shapiro(diff)[1]
    normal = sw_p >= ALPHA

    if normal:
        test_used = "Paired t-test"
        stat_value, p_val = stats.ttest_rel(post, pre)
        eff = cohens_dz(diff)
        eff_label = "Cohen's dz"
    else:
        test_used = "Wilcoxon signed-rank"
        stat_value, p_val = stats.wilcoxon(post, pre, zero_method="wilcox")
        eff = rank_biserial_wilcoxon(diff)
        eff_label = "Rank-biserial r"

    improved = (diff.mean() < 0) if lower_is_better else (diff.mean() > 0)

    return {
        "Test": test_name,
        "Group": group_label,
        "N": n,
        "Mean_Diff": round(diff.mean(), 2),
        "Shapiro_p": round(sw_p, 4),
        "Test_Used": test_used,
        "Statistic": round(stat_value, 3),
        "p_value": p_val,
        "Effect_Size_Type": eff_label,
        "Effect_Size": round(eff, 3),
        "Direction": "Improved" if improved else "Worsened/No change",
    }


# ----------------------------------------------------------------------
# OPTIONAL / SUPPLEMENTARY: full two-way mixed ANOVA (Time x Group)
# ----------------------------------------------------------------------
def run_mixed_anova(df, test_name, pre_col, post_col):
    """
    Runs a full two-way mixed ANOVA (Time = within-subject, Group =
    between-subject) using pingouin. Reports all 3 effects: main effect
    of Time, main effect of Group, and the Time x Group interaction.

    This is a SUPPLEMENTARY analysis, reported alongside the primary
    Welch's t-test / Mann-Whitney U on delta scores (see
    analyze_interaction above). For a strict 2-group x 2-timepoint
    design, the ANOVA's Time x Group interaction p-value is
    mathematically equivalent to the primary delta-based test; the
    ANOVA additionally reports the two main effects and Mauchly's
    sphericity test, which some journals expect to see reported in full.
    """
    sub = df[["subject_id", "Group", pre_col, post_col]].dropna()

    long = pd.melt(
        sub, id_vars=["subject_id", "Group"],
        value_vars=[pre_col, post_col],
        var_name="Time", value_name="Score"
    )
    long["Time"] = long["Time"].map({pre_col: "Pre", post_col: "Post"})

    aov = pg.mixed_anova(
        data=long, dv="Score", within="Time", between="Group",
        subject="subject_id"
    )
    aov.insert(0, "Test", test_name)
    return aov
def _draw_interaction_panel(ax, result, panel_label=None, title_fontsize=10):
    low_pre_post = result["_low_raw"]
    high_pre_post = result["_high_raw"]

    low_means = low_pre_post.mean(axis=0)
    low_sems = low_pre_post.std(axis=0, ddof=1) / np.sqrt(len(low_pre_post))
    high_means = high_pre_post.mean(axis=0)
    high_sems = high_pre_post.std(axis=0, ddof=1) / np.sqrt(len(high_pre_post))

    x = [0, 1]
    ax.errorbar(x, low_means, yerr=low_sems, marker="o", capsize=4,
                color=GROUP_COLORS["Low"], label=f"Low (n={result['N_Low']})", linewidth=2)
    ax.errorbar(x, high_means, yerr=high_sems, marker="s", capsize=4,
                color=GROUP_COLORS["High"], label=f"High (n={result['N_High']})", linewidth=2)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pre", "Post"])
    ax.set_xlim(-0.3, 1.3)

    # --- within-group simple-effect stars (does THIS group change pre->post
    # on its own?), placed just above/below each line's Post-side endpoint,
    # colored to match that group's line. Separate from the interaction
    # test in the title, which asks whether the two groups differ from
    # EACH OTHER in amount of change. ---
    low_simple_p = result.get("Low_simple_p_adj")
    high_simple_p = result.get("High_simple_p_adj")

    if low_simple_p is not None:
        low_stars = sig_stars(low_simple_p)
        ax.text(1.06, low_means[1], low_stars, color=GROUP_COLORS["Low"],
                fontsize=10, fontweight="bold" if low_stars != "ns" else "normal",
                va="center", ha="left")

    if high_simple_p is not None:
        high_stars = sig_stars(high_simple_p)
        ax.text(1.06, high_means[1], high_stars, color=GROUP_COLORS["High"],
                fontsize=10, fontweight="bold" if high_stars != "ns" else "normal",
                va="center", ha="left")

    p_val = result["p_value"]
    stars = sig_stars(result.get("p_adj_FDR", p_val))
    p_report = result.get("p_adj_FDR", p_val)
    p_label = "p_FDR" if "p_adj_FDR" in result else "p"

    title = f"{result['Test']}\nInteraction {p_label} = {p_report:.4f} ({stars})"
    if panel_label:
        ax.text(-0.15, 1.15, panel_label, transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="top", ha="left")
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlim(-0.3, 1.5)  # extra room on the right for the simple-effect star labels


def _jitter(n, width=0.08, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-width, width, n)


def _draw_prism_style_panel(ax, result, panel_label=None, title_fontsize=10):
    """
    Replicates the classic Prism-style 4-column dot plot:
    Low-0m | Low-1m | High-0m | High-1m, individual points + median line,
    with two significance brackets:
      - top bracket: High-1m vs Low-1m (between-group, post-only)
      - lower bracket: High-0m vs High-1m (within-group, High only)
    """
    low_raw = result["_low_raw"]   # columns: [pre, post]
    high_raw = result["_high_raw"]

    positions = {"Low_pre": 0, "Low_post": 1, "High_pre": 2.4, "High_post": 3.4}
    data_map = {
        "Low_pre": low_raw[:, 0], "Low_post": low_raw[:, 1],
        "High_pre": high_raw[:, 0], "High_post": high_raw[:, 1],
    }
    colors = {"Low_pre": GROUP_COLORS["Low"], "Low_post": GROUP_COLORS["Low"],
              "High_pre": GROUP_COLORS["High"], "High_post": GROUP_COLORS["High"]}

    for key, x in positions.items():
        y = data_map[key]
        jitter = _jitter(len(y), seed=hash(key) % 1000)
        ax.scatter(x + jitter, y, color=colors[key], s=18, alpha=0.85, zorder=3)
        median = np.median(y)
        ax.plot([x - 0.18, x + 0.18], [median, median], color=colors[key], linewidth=2, zorder=4)

    ax.set_xticks(list(positions.values()))
    ax.set_xticklabels(["0m", "1m", "0m", "1m"])

    y_all = np.concatenate(list(data_map.values()))
    y_min, y_max = y_all.min(), y_all.max()
    y_range = y_max - y_min if y_max > y_min else 1.0

    # --- lower bracket: High-0m vs High-1m (within-group) ---
    high_p = result.get("High_simple_p_adj")
    if high_p is not None:
        bar_y1 = y_max + 0.06 * y_range
        ax.plot([positions["High_pre"], positions["High_pre"], positions["High_post"], positions["High_post"]],
                 [bar_y1 - 0.015 * y_range, bar_y1, bar_y1, bar_y1 - 0.015 * y_range],
                 color="black", linewidth=1.0)
        ax.text((positions["High_pre"] + positions["High_post"]) / 2, bar_y1 + 0.01 * y_range,
                sig_stars(high_p), ha="center", va="bottom",
                fontsize=13, fontweight="bold")
    else:
        bar_y1 = y_max + 0.06 * y_range

    # --- top bracket: High-1m vs Low-1m (between-group, post only) ---
    post_p = result.get("posttest_p_adj")
    if post_p is not None:
        bar_y2 = bar_y1 + 0.16 * y_range
        ax.plot([positions["Low_post"], positions["Low_post"], positions["High_post"], positions["High_post"]],
                 [bar_y2 - 0.015 * y_range, bar_y2, bar_y2, bar_y2 - 0.015 * y_range],
                 color="black", linewidth=1.0)
        ax.text((positions["Low_post"] + positions["High_post"]) / 2, bar_y2 + 0.01 * y_range,
                sig_stars(post_p), ha="center", va="bottom",
                fontsize=13, fontweight="bold")
        top_y = bar_y2
    else:
        top_y = bar_y1

    ax.set_ylim(y_min - 0.05 * y_range, top_y + 0.18 * y_range)
    ax.set_xlim(-0.5, 3.9)

    if panel_label:
        ax.text(-0.15, 1.15, panel_label, transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="top", ha="left")
    ax.set_title(result["Test"], fontsize=title_fontsize)


def make_prism_style_plot(result, out_path):
    fig, ax = plt.subplots(figsize=(4.5, 5))
    _draw_prism_style_panel(ax, result)
    ax.set_ylabel("Score")
    from matplotlib.lines import Line2D
    legend_elems = [Line2D([0], [0], marker="o", color="w", markerfacecolor=GROUP_COLORS["Low"],
                            markersize=8, label="Low-usage"),
                    Line2D([0], [0], marker="o", color="w", markerfacecolor=GROUP_COLORS["High"],
                            markersize=8, label="High-usage")]
    ax.legend(handles=legend_elems, loc="best", fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_combined_prism_figure(results, out_path):
    n_tests = len(results)
    n_cols = 3
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 5 * n_rows))
    axes = axes.flatten()
    panel_labels = [chr(ord("A") + i) for i in range(n_tests)]

    for ax, result, label in zip(axes, results, panel_labels):
        _draw_prism_style_panel(ax, result, panel_label=label)

    for ax in axes[n_tests:]:
        ax.axis("off")

    axes[0].set_ylabel("Score")
    axes[n_cols].set_ylabel("Score")

    from matplotlib.lines import Line2D
    legend_elems = [Line2D([0], [0], marker="o", color="w", markerfacecolor=GROUP_COLORS["Low"],
                            markersize=8, label="Low-usage"),
                    Line2D([0], [0], marker="o", color="w", markerfacecolor=GROUP_COLORS["High"],
                            markersize=8, label="High-usage")]
    fig.legend(handles=legend_elems, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.02), frameon=False)

    fig.suptitle(
        "Lower bracket = within-High-group change (0m vs 1m). "
        "Upper bracket = between-group difference at post (Low-1m vs High-1m).",
        fontsize=8, y=1.02
    )
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_interaction_plot(result, out_path):
    fig, ax = plt.subplots(figsize=(4.5, 5))
    _draw_interaction_panel(ax, result)
    ax.set_ylabel("Score")
    ax.legend(loc="best", fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_combined_interaction_figure(results, out_path):
    n_tests = len(results)
    n_cols = 3
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 5 * n_rows))
    axes = axes.flatten()
    panel_labels = [chr(ord("A") + i) for i in range(n_tests)]

    for ax, result, label in zip(axes, results, panel_labels):
        _draw_interaction_panel(ax, result, panel_label=label)

    # hide any unused subplot slots (5 tests into a 2x3 grid leaves 1 empty)
    for ax in axes[n_tests:]:
        ax.axis("off")

    axes[0].set_ylabel("Score")
    axes[n_cols].set_ylabel("Score")  # first axis of the second row
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.02), frameon=False)

    fig.suptitle(
        "Stars in title = Group x Time interaction (Low vs High change compared to each other). "
        "Stars beside each Post point = that group's own pre->post change (colored to match its line).",
        fontsize=8, y=1.02
    )
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(csv_path):
    df = pd.read_csv(csv_path)
    if "Group" not in df.columns:
        raise ValueError("CSV must contain a 'Group' column with values 'Low'/'High'.")
    df["Group"] = df["Group"].str.strip().str.title()  # normalize e.g. 'low' -> 'Low'

    # ---- PRIMARY: interaction analysis ----
    interaction_results = []
    for test_name, cfg in TESTS.items():
        if cfg["pre"] not in df.columns or cfg["post"] not in df.columns:
            print(f"WARNING: columns for {test_name} not found, skipping.")
            continue
        res = analyze_interaction(df, test_name, cfg["pre"], cfg["post"], cfg["lower_is_better"])
        interaction_results.append(res)

    p_adj = fdr_bh([r["p_value"] for r in interaction_results])
    for r, pa in zip(interaction_results, p_adj):
        r["p_adj_FDR"] = round(pa, 4)
        r["Significant_after_FDR"] = pa < ALPHA

    interaction_export = pd.DataFrame(
        [{k: v for k, v in r.items() if not k.startswith("_")} for r in interaction_results]
    )
    interaction_export["p_value"] = interaction_export["p_value"].round(4)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print("\n=== PRIMARY: Group (Low vs High) x Time Interaction ===\n")
    print(interaction_export.to_string(index=False))
    interaction_export.to_csv("interaction_summary.csv", index=False)
    print("\nSaved: interaction_summary.csv")

    # ---- SECONDARY: simple effects (within-group paired pre-post) ----
    simple_results = []
    for group_label in GROUP_ORDER:
        df_group = df[df["Group"] == group_label]
        for test_name, cfg in TESTS.items():
            if cfg["pre"] not in df.columns or cfg["post"] not in df.columns:
                continue
            res = analyze_simple_effect(df_group, group_label, test_name,
                                         cfg["pre"], cfg["post"], cfg["lower_is_better"])
            simple_results.append(res)

    # FDR applied within each group's family of 5 tests separately
    simple_df = pd.DataFrame(simple_results)
    simple_df["p_adj_FDR"] = np.nan
    for group_label in GROUP_ORDER:
        mask = simple_df["Group"] == group_label
        simple_df.loc[mask, "p_adj_FDR"] = fdr_bh(simple_df.loc[mask, "p_value"].to_numpy())
    simple_df["p_adj_FDR"] = simple_df["p_adj_FDR"].round(4)
    simple_df["Significant_after_FDR"] = simple_df["p_adj_FDR"] < ALPHA
    simple_df["p_value"] = simple_df["p_value"].round(4)

    print("\n=== SECONDARY: Within-Group Simple Effects (Pre vs Post) ===\n")
    print(simple_df.to_string(index=False))
    simple_df.to_csv("simple_effects_summary.csv", index=False)
    print("\nSaved: simple_effects_summary.csv")

    # attach each test's Low/High simple-effect adjusted p-value onto the
    # interaction_results dicts, so the plots can annotate both the
    # interaction significance (title) and each group's own pre-post
    # significance (stars next to each line's Post point)
    simple_lookup = simple_df.set_index(["Test", "Group"])["p_adj_FDR"].to_dict()
    for r in interaction_results:
        r["Low_simple_p_adj"] = simple_lookup.get((r["Test"], "Low"))
        r["High_simple_p_adj"] = simple_lookup.get((r["Test"], "High"))

    # ---- SUPPLEMENTARY: between-group comparison at each timepoint ----
    # (baseline check + the "post-only" comparison used in the Prism-style plot)
    between_results = []
    for test_name, cfg in TESTS.items():
        if cfg["pre"] not in df.columns or cfg["post"] not in df.columns:
            continue
        between_results.append(analyze_between_group_at_timepoint(df, test_name, cfg["pre"], "0m (baseline)"))
        between_results.append(analyze_between_group_at_timepoint(df, test_name, cfg["post"], "1m (post)"))

    between_df = pd.DataFrame(between_results)
    between_df["p_adj_FDR"] = np.nan
    for tp in ["0m (baseline)", "1m (post)"]:
        mask = between_df["Timepoint"] == tp
        between_df.loc[mask, "p_adj_FDR"] = fdr_bh(between_df.loc[mask, "p_value"].to_numpy())
    between_df["p_adj_FDR"] = between_df["p_adj_FDR"].round(4)
    between_df["Significant_after_FDR"] = between_df["p_adj_FDR"] < ALPHA
    between_df["p_value"] = between_df["p_value"].round(4)

    print("\n=== SUPPLEMENTARY: Between-Group Comparison at Each Timepoint ===\n")
    print("Baseline row lets you confirm Low and High did not already differ before")
    print("the intervention. The 1m (post) row is the comparison shown as the top")
    print("bracket in Prism-style plots (Low-1m vs High-1m).\n")
    print(between_df.to_string(index=False))
    between_df.to_csv("between_group_timepoint_summary.csv", index=False)
    print("\nSaved: between_group_timepoint_summary.csv")

    # attach the post-only between-group p-value for the Prism-style plot's top bracket
    posttest_lookup = between_df[between_df["Timepoint"] == "1m (post)"].set_index("Test")["p_adj_FDR"].to_dict()
    for r in interaction_results:
        r["posttest_p_adj"] = posttest_lookup.get(r["Test"])

    # ---- SUPPLEMENTARY: full two-way mixed ANOVA (Time x Group) ----
    anova_tables = []
    for test_name, cfg in TESTS.items():
        if cfg["pre"] not in df.columns or cfg["post"] not in df.columns:
            continue
        aov = run_mixed_anova(df, test_name, cfg["pre"], cfg["post"])
        anova_tables.append(aov)

    anova_full = pd.concat(anova_tables, ignore_index=True)
    print("\n=== SUPPLEMENTARY: Full Two-Way Mixed ANOVA (Time x Group) ===\n")
    print("Note: the 'Interaction' row is conceptually equivalent to the primary")
    print("Welch/Mann-Whitney test above for this 2x2 design, but p-values may differ")
    print("slightly: ANOVA's F-test assumes homogeneous variance across groups and runs")
    print("on the raw scale regardless of normality, while the primary test above chose")
    print("Welch's correction (unequal variance) or Mann-Whitney (non-normal) per test.")
    print("Main effects of Time and Group are reported here for completeness, as some")
    print("journals expect the full ANOVA table.\n")
    print(anova_full.to_string(index=False))
    anova_full.to_csv("mixed_anova_summary.csv", index=False)
    print("\nSaved: mixed_anova_summary.csv")

    # ---- Plots ----
    for r in interaction_results:
        fname = f"interaction_plot_{r['Test'].replace(' ', '_').replace('-', '')}.png"
        make_interaction_plot(r, fname)
        print(f"Saved: {fname}")

    make_combined_interaction_figure(interaction_results, "combined_interaction_panel.png")
    print("Saved: combined_interaction_panel.png")

    for r in interaction_results:
        fname = f"prism_plot_{r['Test'].replace(' ', '_').replace('-', '')}.png"
        make_prism_style_plot(r, fname)
        print(f"Saved: {fname}")

    make_combined_prism_figure(interaction_results, "combined_prism_panel.png")
    print("Saved: combined_prism_panel.png")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python prepost_group_comparison.py your_data.csv")
        sys.exit(1)
    main(sys.argv[1])