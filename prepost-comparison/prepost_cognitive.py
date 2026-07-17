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
- Saves estimation plots per test (estimation_plot_<test>.png)
- Saves combined estimation figure (combined_estimation_all_tests.png)

Estimation plots (Gardner-Altman / Cumming style) added 2026-07:
  make_estimation_plot()            -- single-test version
  make_combined_estimation_figure() -- 5-panel manuscript figure

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


def _bootstrap_dist_from_diff(diff, n_boot=N_BOOT):
    """
    Returns the full bootstrap resampling distribution of the mean of diff.
    Computed ONCE per test (inside analyze_one_test) and reused for both the
    reported CI and the Gardner-Altman violin, so the two always agree
    exactly instead of being drawn from two separately-seeded resamples.
    """
    diff = np.asarray(diff)
    n = len(diff)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        boot[i] = RNG.choice(diff, size=n, replace=True).mean()
    return boot


def percentile_ci(boot_dist, alpha=ALPHA):
    """Percentile 95% CI computed FROM an already-generated bootstrap distribution."""
    lo = np.percentile(boot_dist, 100 * (alpha / 2))
    hi = np.percentile(boot_dist, 100 * (1 - alpha / 2))
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

    boot_dist = _bootstrap_dist_from_diff(diff)  # single resample, reused for CI + violin
    ci_lo, ci_hi = percentile_ci(boot_dist)

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
        "_boot_dist_raw": boot_dist,  # reused by GA plots so violin matches reported CI exactly
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
    ax.scatter(np.zeros(n), pre, color="#4C72B0", zorder=3, s=28, label="Pre")
    ax.scatter(np.ones(n), post, color="#DD8452", zorder=3, s=28, label="Post")
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


# ──────────────────────────────────────────────────────────────────────────────
# Estimation plots — two publication-quality styles
#
#   Gardner-Altman (1986 / Ho 2019):
#     Raw data (left axis) + effect-size axis (right, SAME y-scale, connected).
#     The right axis is shifted vertically so that 0 aligns with the Pre mean
#     (the baseline / reference group, per standard Gardner-Altman convention).
#     Half-violin of bootstrap distribution opens to the RIGHT.
#
#   Cumming (2012 / Ho 2019):
#     Raw data (top sub-panel) + effect-size axis (bottom sub-panel, centred at 0).
#     The two sub-panels share the same x positions so data dots are vertically
#     aligned with the effect dot below.
#
# Key design rules (following dabest / Nature Methods conventions):
#   • Slopegraph lines connect every paired observation.
#   • Mean ± SD shown as "gapped" vertical bars (gap = white space around mean).
#   • Effect-size panel: half-violin (bootstrap KDE) + thick CI whisker + mean dot.
#   • Zero reference: dashed horizontal line at diff = 0.
#   • ALL annotation text (stars, CI, effect size) goes into a clean text-box
#     BELOW the violin (Cumming) or to the far right (Gardner-Altman), never
#     overlapping the data cloud.
# ──────────────────────────────────────────────────────────────────────────────

from scipy.stats import gaussian_kde  # imported here so helpers below can use it


def _gapped_meansem(ax, xpos, vals, color, dot_size=55):
    """
    Draw a dabest-style 'gapped line': mean ± 1 SEM with a white gap around
    the mean dot, so the mean is clearly visible.
    """
    m   = vals.mean()
    sem = vals.std(ddof=1) / np.sqrt(len(vals))
    gap = sem * 0.30
    ax.plot([xpos, xpos], [m - sem, m - gap], color=color, lw=2.5,
            solid_capstyle="butt", zorder=5)
    ax.plot([xpos, xpos], [m + gap, m + sem], color=color, lw=2.5,
            solid_capstyle="butt", zorder=5)
    ax.scatter([xpos], [m], color=color, s=dot_size, zorder=6)


def _draw_raw_slopegraph(ax, pre, post, scatter_alpha=0.85, line_alpha=0.18):
    """
    Draw jittered dots + slopegraph lines on ax.
    Pre at x=0, Post at x=1.
    Returns handles for legend.
    """
    n = len(pre)
    jk = 0.07
    jx_pre  = RNG.uniform(-jk, jk, n)
    jx_post = RNG.uniform(-jk, jk, n)

    for i in range(n):
        ax.plot([jx_pre[i], 1 + jx_post[i]], [pre[i], post[i]],
                color="#888888", alpha=line_alpha, lw=0.6, zorder=1)

    h1 = ax.scatter(jx_pre,     pre,  color="#4C72B0", s=28, alpha=scatter_alpha,
                    zorder=3, label="Pre")
    h2 = ax.scatter(1 + jx_post, post, color="#DD8452", s=28, alpha=scatter_alpha,
                    zorder=3, label="Post")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pre", "Post"], fontsize=9)
    ax.set_xlim(-0.50, 1.50)
    ax.tick_params(axis="y", labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    return h1, h2


def _draw_effect_violin(ax, boot_means, mean_d, ci_lo, ci_hi,
                        x_anchor=0.0, violin_dir=+1,
                        violin_scale=0.30, color="#4C5D73"):
    """
    Draw half-violin (bootstrap KDE) + CI whisker + mean dot.

    x_anchor   : x-position of the CI/dot (e.g. 0 for Gardner-Altman, 0 for Cumming)
    violin_dir : +1 = violin opens to the RIGHT, -1 = opens to the LEFT
    """
    bmin, bmax = boot_means.min(), boot_means.max()
    spread = max(bmax - bmin, 1e-6)
    kde_y = np.linspace(bmin - 0.25 * spread, bmax + 0.25 * spread, 400)
    try:
        kde_v = gaussian_kde(boot_means, bw_method="scott")(kde_y)
        kde_v = kde_v / kde_v.max() * violin_scale
        ax.fill_betweenx(kde_y,
                         x_anchor,
                         x_anchor + violin_dir * kde_v,
                         color=color, alpha=0.20, linewidth=0)
        ax.plot(x_anchor + violin_dir * kde_v, kde_y,
                color=color, alpha=0.85, linewidth=1.0)
    except Exception:
        pass

    # CI whisker (thick line)
    ax.plot([x_anchor, x_anchor], [ci_lo, ci_hi],
            color=color, lw=2.2, solid_capstyle="round", zorder=4)
    # mean diff dot
    ax.scatter([x_anchor], [mean_d], color=color, s=65, zorder=5)
    # zero reference
    ax.axhline(0, color="black", lw=0.9, ls="--", alpha=0.85, zorder=2)


def _annotation_box(ax, result, fontsize=7, loc="lower right"):
    """
    Place a compact annotation text-box with mean diff, CI, effect size.
    Significance stars are NOT repeated here -- they're already shown as a
    bracket above the raw (left) panel, so this box only adds the numeric
    detail that bracket doesn't carry.
    loc: 'lower right' | 'upper right' | 'lower left' | 'upper left'
    """
    md     = result["Mean_Diff"]
    ci_lo  = result["Boot_95CI_Low"]
    ci_hi  = result["Boot_95CI_High"]
    es     = result["Effect_Size"]
    eslbl  = result["Effect_Size_Type"]

    txt = (f"Δ = {md:+.2f}\n"
           f"95 % CI [{ci_lo:.2f}, {ci_hi:.2f}]\n"
           f"{eslbl} = {es:.3f}")

    loc_map = {
        "lower right": (0.97, 0.03, "right",  "bottom"),
        "upper right": (0.97, 0.97, "right",  "top"),
        "lower left":  (0.03, 0.03, "left",   "bottom"),
        "upper left":  (0.03, 0.97, "left",   "top"),
    }
    x, y, ha, va = loc_map.get(loc, loc_map["lower right"])
    ax.text(x, y, txt, transform=ax.transAxes,
            fontsize=fontsize, ha=ha, va=va, linespacing=1.6,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                      edgecolor="none", alpha=0.90, linewidth=0.8))


# ── Gardner-Altman ──────────────────────────────────────────────────────────

def _ga_panel(ax_raw, ax_eff, result, boot_means,
              panel_label=None, title_fontsize=10):
    """
    One Gardner-Altman panel.

    ax_raw : left axes — raw slopegraph
    ax_eff : right axes — effect-size axis
               * shares the SAME y-scale as ax_raw
               * y is translated so that 0 aligns with Pre mean (baseline)
    """
    pre    = result["_pre_raw"]
    post   = result["_post_raw"]
    mean_d = result["Mean_Diff"]
    ci_lo  = result["Boot_95CI_Low"]
    ci_hi  = result["Boot_95CI_High"]
    n      = result["N"]

    # --- raw panel ---
    h1, h2 = _draw_raw_slopegraph(ax_raw, pre, post)
    stars = sig_stars(result["p_adj_FDR"])
    
    if stars != "ns":
        ymin, ymax = ax_raw.get_ylim()
        yrange = ymax - ymin
        bar_y = ymax - 0.02 * yrange
        tick_h = 0.02 * yrange

        ax_raw.plot(
            [0, 0, 1, 1],
            [bar_y - tick_h, bar_y, bar_y, bar_y - tick_h],
            color="black",
            lw=1.2,
            clip_on=False,
        )

        ax_raw.text(
            0.5,
            bar_y + 0.012 * yrange,
            stars,
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

        ax_raw.set_ylim(ymin, ymax + 0.08 * yrange)   

    ax_raw.set_ylabel("Score", fontsize=9)

    test   = result["Test"]
    method = result["Test_Used"]
    p_adj  = result["p_adj_FDR"]
    ax_raw.set_title(
        f"{test}  (n = {n})\n{method},  p_FDR = {p_adj:.4f}",
        fontsize=title_fontsize, pad=5,
    )
    if panel_label:
        ax_raw.text(-0.20, 1.12, panel_label, transform=ax_raw.transAxes,
                    fontsize=14, fontweight="bold", va="top", ha="left")

    # --- effect axis ---
    # Gardner-Altman convention (Ho et al. 2019, Nat Methods): the effect-size
    # axis zero aligns with the mean of the REFERENCE/baseline group. In a
    # paired pre-post design, Pre is the baseline, so 0 aligns with Pre-mean
    # (not Post-mean) -- this lets the effect dot be read directly as "how
    # far above/below the starting point" on the same visual scale.
    pre_mean = pre.mean()
    raw_ylim  = ax_raw.get_ylim()

    # Transform: effect_y = raw_y - pre_mean  ->  raw_y = effect_y + pre_mean
    eff_lo = raw_ylim[0] - pre_mean
    eff_hi = raw_ylim[1] - pre_mean
    ax_eff.set_ylim(eff_lo, eff_hi)

    _draw_effect_violin(ax_eff, boot_means, mean_d, ci_lo, ci_hi,
                        x_anchor=0.0, violin_dir=+1)

    # tick labels on the right
    ax_eff.yaxis.set_label_position("right")
    ax_eff.yaxis.tick_right()
    ax_eff.tick_params(axis="y", labelsize=8)
    ax_eff.set_xticks([])
    ax_eff.set_xlim(-0.55, 0.85)
    ax_eff.spines[["top", "left", "bottom"]].set_visible(False)
    ax_eff.set_ylabel("Mean difference", fontsize=8, labelpad=6)

    # Annotation box — bottom of effect axis (avoids violin which opens right)
    _annotation_box(ax_eff, result, fontsize=7, loc="lower right")

    return h1, h2


def make_ga_plot(result, out_path):
    """
    Single-test Gardner-Altman estimation plot.
    Raw data on the LEFT (wider); effect-size axis on the RIGHT.
    The two axes share the same y-scale; the right axis is shifted so 0 = Pre mean.
    """
    boot_means = result["_boot_dist_raw"]  # same resample used for the reported CI (no re-draw)

    fig = plt.figure(figsize=(6.0, 4.8))
    ax_raw = fig.add_axes([0.12, 0.13, 0.52, 0.74])
    ax_eff = fig.add_axes([0.68, 0.13, 0.26, 0.74])

    h1, h2 = _ga_panel(ax_raw, ax_eff, result, boot_means, title_fontsize=11)

    fig.legend([h1, h2], ["Pre", "Post"],
               loc="lower center", ncol=2, fontsize=8,
               frameon=False, bbox_to_anchor=(0.38, -0.01))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_combined_ga_figure(results, out_path):
    """
    Combined Gardner–Altman figure.

                A (MMSE)

        B (TMT-A)      C (TMT-B)

        D (DSF)        E (DSB)
    """

    import matplotlib.gridspec as gridspec

    boot_dists = [r["_boot_dist_raw"] for r in results]  # reuse, same resample as reported CI
    panel_labels = [chr(ord("A") + i) for i in range(len(results))]

    fig = plt.figure(figsize=(12, 14))

    outer = gridspec.GridSpec(
        3, 2,
        figure=fig,
        height_ratios=[1.15, 1, 1],
        hspace=0.45,
        wspace=0.28,
        left=0.06,
        right=0.97,
        top=0.96,
        bottom=0.07,
    )

    panel_positions = [
        outer[0, :],     # MMSE
        outer[1, 0],     # TMT-A
        outer[1, 1],     # TMT-B
        outer[2, 0],     # DSF
        outer[2, 1],     # DSB
    ]

    # Reference x-coordinates for a single-column cell's raw/eff split (using
    # column 0, row 1 -- i.e. panel B's cell -- as the template). Panel A will
    # be placed at EXACTLY these x-boundaries, rather than re-deriving them
    # via a ratio split of the double-width row cell (which was subtly wrong:
    # it silently absorbed half of the inter-column wspace gap into the
    # content width, making A ~14% wider than B/D and shifting its "A" label).
    _ref_inner = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[1, 0], width_ratios=[3.4, 0.9], wspace=0.02,
    )
    _ref_raw_bbox = _ref_inner[0].get_position(fig)
    _ref_eff_bbox = _ref_inner[1].get_position(fig)

    for result, label, boot, cell in zip(
            results,
            panel_labels,
            boot_dists,
            panel_positions):

        if label == "A":
            # Use column-0's exact x-range (from the reference split above),
            # combined with panel A's own row's y-range, so A's raw/eff axes
            # are pixel-identical in width and x-position to B/D's.
            row_bbox = cell.get_position(fig)
            y0, y1 = row_bbox.y0, row_bbox.y1
            ax_raw = fig.add_axes([_ref_raw_bbox.x0, y0,
                                    _ref_raw_bbox.width, y1 - y0])
            ax_eff = fig.add_axes([_ref_eff_bbox.x0, y0,
                                    _ref_eff_bbox.width, y1 - y0])
        else:
            inner = gridspec.GridSpecFromSubplotSpec(
                1, 2, subplot_spec=cell,
                width_ratios=[3.4, 0.9], wspace=0.02,
            )
            ax_raw = fig.add_subplot(inner[0])
            ax_eff = fig.add_subplot(inner[1])

        _ga_panel(
            ax_raw,
            ax_eff,
            result,
            boot,
            panel_label=label,
            title_fontsize=10,
        )

    handles = [
        plt.scatter([], [], color="#4C72B0", s=30, label="Pre"),
        plt.scatter([], [], color="#DD8452", s=30, label="Post"),
    ]

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )

    fig.savefig(
        out_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# Keep old names as aliases so main() can call both
def make_estimation_plot(result, out_path):
    """Alias → Gardner-Altman (kept for backward compatibility)."""
    make_ga_plot(result, out_path)


def make_combined_estimation_figure(results, out_path):
    """Alias → Gardner-Altman combined figure (kept for backward compatibility)."""
    make_combined_ga_figure(results, out_path)


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

    # ── Gardner-Altman estimation plots ───────────────────────────────────────
    print("\n── Gardner-Altman estimation plots ──")
    for r in results:
        safe = r["Test"].replace(" ", "_").replace("-", "")
        fname = f"ga_plot_{safe}.png"
        make_ga_plot(r, fname)
        print(f"Saved: {fname}")

    make_combined_ga_figure(results, "combined_ga_all_tests.png")
    print("Saved: combined_ga_all_tests.png")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python prepost_cognitive_analysis.py your_data.csv")
        sys.exit(1)
    main(sys.argv[1])