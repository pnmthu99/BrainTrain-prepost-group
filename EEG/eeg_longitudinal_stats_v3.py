# -*- coding: utf-8 -*-
"""
EEG longitudinal statistics for Pre, Post, and 3-month follow-up.

Input is the wide table produced by aggregate_features.py, with one row per
subject and columns named:
    {task}_{ROI}_{feature}_{pre|post|3m}

For each task x ROI x feature, this script:
  1. Runs a Friedman omnibus test across Pre, Post, and 3m.
  2. Runs the three planned paired contrasts with Wilcoxon signed-rank tests:
       Post - Pre, 3m - Pre, and 3m - Post.
  3. Applies Benjamini-Hochberg FDR separately within each task x feature
     family. Omnibus tests and contrasts are corrected separately; contrast
     correction includes every reported contrast in that family.

Friedman and the paired contrasts use subjects with all three measurements for
the feature. This makes the analysis robust to non-normal distributions, but
does not use partially observed subjects. If missingness is material, use a
linear mixed-effects model instead.

Usage:
    python eeg_longitudinal_stats_v3.py wide_feature_table.csv

All CSV and PNG outputs are written to results/ in the current working directory.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ALPHA = 0.05
N_BOOT = 5000
RNG = np.random.default_rng(42)

TASKS = ["3B", "4", "5", "6"]
ROIS = ["Frontal", "Central", "Parietal", "Temporal", "Occipital"]
BANDS = ["delta", "theta", "alpha", "beta"]
TIMEPOINTS = ["pre", "post", "3m"]
CONTRASTS = [
    ("post_minus_pre", "post", "pre"),
    ("3m_minus_pre", "3m", "pre"),
    ("3m_minus_post", "3m", "post"),
]
FEATURE_FAMILIES = {
    "band_power": [f"{band}_relpower" for band in BANDS],
    "theta_alpha_ratio": ["theta_alpha_ratio"],
    "theta_beta_ratio": ["theta_beta_ratio"],
    "entropy": ["entropy"],
}


def fdr_bh(p_values):
    """Return Benjamini-Hochberg adjusted p-values, retaining NaNs."""
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p_values), np.nan)
    valid = ~np.isnan(p_values)
    if not valid.any():
        return adjusted

    values = p_values[valid]
    order = np.argsort(values)
    ranked = values[order]
    n_values = len(values)
    corrected = ranked * n_values / np.arange(1, n_values + 1)
    corrected = np.minimum.accumulate(corrected[::-1])[::-1]
    values_adjusted = np.empty(n_values)
    values_adjusted[order] = np.clip(corrected, 0, 1)
    adjusted[valid] = values_adjusted
    return adjusted


def sig_stars(p_value):
    if pd.isna(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < ALPHA:
        return "*"
    return ""


def rank_biserial_wilcoxon(difference):
    difference = np.asarray(difference)
    difference = difference[difference != 0]
    if len(difference) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(difference))
    positive = ranks[difference > 0].sum()
    negative = ranks[difference < 0].sum()
    return (positive - negative) / (positive + negative)


def bootstrap_ci_mean(values, n_boot=N_BOOT, alpha=ALPHA):
    values = np.asarray(values, dtype=float)
    indices = RNG.integers(0, len(values), size=(n_boot, len(values)))
    boot_means = values[indices].mean(axis=1)
    return (
        np.percentile(boot_means, 100 * alpha / 2),
        np.percentile(boot_means, 100 * (1 - alpha / 2)),
    )


def paired_contrast(values_a, values_b):
    """Summarize and test values_a - values_b for matched observations."""
    difference = values_a - values_b
    mean_difference = difference.mean()
    ci_low, ci_high = bootstrap_ci_mean(difference)

    if np.all(difference == 0):
        statistic, p_value, effect = 0.0, 1.0, 0.0
    else:
        statistic, p_value = stats.wilcoxon(
            values_a, values_b, alternative="two-sided", zero_method="wilcox"
        )
        effect = rank_biserial_wilcoxon(difference)

    return {
        "N_complete": len(difference),
        "Mean_A": values_a.mean(),
        "Mean_B": values_b.mean(),
        "Mean_Diff": mean_difference,
        "Boot_95CI_Low": ci_low,
        "Boot_95CI_High": ci_high,
        "Test_Used": "Wilcoxon signed-rank",
        "Statistic": statistic,
        "p_value": p_value,
        "Effect_Size_Type": "Rank-biserial r",
        "Effect_Size": effect,
    }


def analyze_feature(df, columns):
    complete = df[list(columns.values())].dropna()
    if len(complete) < 3:
        return None, []

    values = {timepoint: complete[column].to_numpy(dtype=float)
              for timepoint, column in columns.items()}
    stacked = np.column_stack([values[timepoint] for timepoint in TIMEPOINTS])

    if np.all(np.ptp(stacked, axis=1) == 0):
        friedman_statistic, friedman_p = 0.0, 1.0
    else:
        friedman_statistic, friedman_p = stats.friedmanchisquare(
            *(values[timepoint] for timepoint in TIMEPOINTS)
        )

    omnibus = {
        "N_complete": len(complete),
        "Mean_Pre": values["pre"].mean(),
        "Mean_Post": values["post"].mean(),
        "Mean_3m": values["3m"].mean(),
        "Test_Used": "Friedman test",
        "Statistic": friedman_statistic,
        "p_value": friedman_p,
    }

    contrasts = []
    for contrast_name, numerator, denominator in CONTRASTS:
        contrast = paired_contrast(values[numerator], values[denominator])
        contrast.update({
            "Contrast": contrast_name,
            "Timepoint_A": numerator,
            "Timepoint_B": denominator,
        })
        contrasts.append(contrast)
    return omnibus, contrasts


def add_fdr(results, p_column="p_value", output_column="p_adj_FDR"):
    results[output_column] = fdr_bh(results[p_column].to_numpy())
    results["Significant_after_FDR"] = results[output_column] < ALPHA
    return results


def run_all_tasks(df):
    omnibus_by_task = {}
    contrasts_by_task = {}

    for task in TASKS:
        task_omnibus, task_contrasts = [], []
        for family_name, measures in FEATURE_FAMILIES.items():
            family_omnibus, family_contrasts = [], []
            for roi in ROIS:
                for measure in measures:
                    base = f"{task}_{roi}_{measure}"
                    columns = {timepoint: f"{base}_{timepoint}" for timepoint in TIMEPOINTS}
                    if not all(column in df.columns for column in columns.values()):
                        continue

                    omnibus, contrasts = analyze_feature(df, columns)
                    if omnibus is None:
                        continue

                    metadata = {
                        "Task": task,
                        "Family": family_name,
                        "ROI": roi,
                        "Measure": measure,
                    }
                    omnibus.update(metadata)
                    family_omnibus.append(omnibus)
                    for contrast in contrasts:
                        contrast.update(metadata)
                        family_contrasts.append(contrast)

            if family_omnibus:
                task_omnibus.append(add_fdr(pd.DataFrame(family_omnibus)))
                task_contrasts.append(add_fdr(pd.DataFrame(family_contrasts)))

        if task_omnibus:
            omnibus_by_task[task] = pd.concat(task_omnibus, ignore_index=True)
            contrasts_by_task[task] = pd.concat(task_contrasts, ignore_index=True)
        else:
            print(f"WARNING: no valid Pre/Post/3m features found for task {task}")

    return omnibus_by_task, contrasts_by_task


def make_contrast_heatmap(task_df, task, family_name, measures, contrast_name, out_path):
    subset = task_df[
        (task_df["Family"] == family_name) & (task_df["Contrast"] == contrast_name)
    ]
    if subset.empty:
        return False

    effects = subset.pivot(index="ROI", columns="Measure", values="Effect_Size")
    q_values = subset.pivot(index="ROI", columns="Measure", values="p_adj_FDR")
    effects = effects.reindex(index=ROIS, columns=measures)
    q_values = q_values.reindex(index=ROIS, columns=measures)

    values = effects.to_numpy(dtype=float)
    max_effect = np.nanmax(np.abs(values)) if not np.all(np.isnan(values)) else 1.0
    fig, ax = plt.subplots(figsize=(max(5.5, 1.8 * len(measures)), 5))
    image = ax.imshow(values, cmap="RdBu_r", vmin=-max_effect, vmax=max_effect, aspect="auto")
    ax.set_xticks(range(len(measures)))
    ax.set_xticklabels(measures, rotation=45, ha="right")
    ax.set_yticks(range(len(ROIS)))
    ax.set_yticklabels(ROIS)

    for row in range(len(ROIS)):
        for column in range(len(measures)):
            q_value = q_values.to_numpy()[row, column]
            if pd.notna(q_value) and q_value < ALPHA:
                ax.text(
                    column, row, f"{sig_stars(q_value)}\nq={q_value:.3f}",
                    ha="center", va="center", color="white", fontweight="bold", fontsize=9,
                )

    ax.set_title(
        f"Task {task} - {family_name} - {contrast_name}\n"
        "Color = rank-biserial r; text = FDR-significant contrast"
    )
    fig.colorbar(image, ax=ax, label="Rank-biserial r", shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def make_contrast_dotplot(task_df, task, family_name, contrast_name, out_path):
    subset = task_df[
        (task_df["Family"] == family_name) & (task_df["Contrast"] == contrast_name)
    ]
    if subset.empty:
        return False
    subset = subset.set_index("ROI").reindex(ROIS)

    fig, ax = plt.subplots(figsize=(6, 4))
    for position, roi in enumerate(ROIS):
        row = subset.loc[roi]
        if pd.isna(row["Mean_Diff"]):
            continue
        significant = row["p_adj_FDR"] < ALPHA
        color = "#C1272D" if row["Mean_Diff"] > 0 else "#1F77B4"
        if not significant:
            color = "lightgray"
        ax.errorbar(
            row["Mean_Diff"], position,
            xerr=[[row["Mean_Diff"] - row["Boot_95CI_Low"]],
                  [row["Boot_95CI_High"] - row["Mean_Diff"]]],
            fmt="o", color=color, ecolor=color, capsize=4, markersize=8,
        )
        if significant:
            ax.text(row["Boot_95CI_High"], position,
                    f"  {sig_stars(row['p_adj_FDR'])} q={row['p_adj_FDR']:.3f}",
                    va="center", fontsize=9)

    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_yticks(range(len(ROIS)))
    ax.set_yticklabels(ROIS)
    ax.set_xlabel("Mean difference with 95% bootstrap CI")
    ax.set_title(f"Task {task} - {family_name} - {contrast_name}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def make_trajectory_plot(df, row, out_path):
    base = f"{row.Task}_{row.ROI}_{row.Measure}"
    columns = [f"{base}_{timepoint}" for timepoint in TIMEPOINTS]
    values = df[columns].dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return False

    fig, ax = plt.subplots(figsize=(5.5, 4))
    x_positions = np.arange(len(TIMEPOINTS))
    ax.plot(x_positions, values.T, color="lightgray", linewidth=0.8, alpha=0.7)
    means = values.mean(axis=0)
    intervals = np.array([bootstrap_ci_mean(values[:, index]) for index in range(len(TIMEPOINTS))])
    ax.errorbar(
        x_positions, means,
        yerr=[means - intervals[:, 0], intervals[:, 1] - means],
        fmt="o-", color="black", capsize=5, linewidth=2, label="Mean with 95% bootstrap CI",
    )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(["Pre", "Post", "3m"])
    ax.set_ylabel(row.Measure)
    ax.set_title(f"Task {row.Task} - {row.ROI} - {row.Measure}\nFriedman q={row.p_adj_FDR:.3f}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def rounded_for_export(results):
    results = results.copy()
    numeric_columns = results.select_dtypes(include=np.number).columns
    results[numeric_columns] = results[numeric_columns].round(4)
    return results


def main(csv_path):
    df = pd.read_csv(csv_path)
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    print(f"Loaded {csv_path}: {len(df)} subjects, {len(df.columns)} columns")

    omnibus_by_task, contrasts_by_task = run_all_tasks(df)
    if not omnibus_by_task:
        raise ValueError("No features with valid Pre, Post, and 3m columns were found.")

    omnibus_summary = pd.concat(omnibus_by_task.values(), ignore_index=True)
    contrasts_summary = pd.concat(contrasts_by_task.values(), ignore_index=True)
    omnibus_summary_path = output_dir / "eeg_longitudinal_omnibus_summary.csv"
    contrasts_summary_path = output_dir / "eeg_longitudinal_contrasts_summary.csv"
    rounded_for_export(omnibus_summary).to_csv(omnibus_summary_path, index=False)
    rounded_for_export(contrasts_summary).to_csv(contrasts_summary_path, index=False)
    print(f"Saved omnibus summary: {omnibus_summary_path} ({len(omnibus_summary)} rows)")
    print(f"Saved contrast summary: {contrasts_summary_path} ({len(contrasts_summary)} rows)")

    omnibus_significant = omnibus_summary[omnibus_summary["Significant_after_FDR"]]
    print(f"Omnibus results significant after FDR: {len(omnibus_significant)}/{len(omnibus_summary)}")
    print(f"Contrasts significant after FDR: {contrasts_summary['Significant_after_FDR'].sum()}/{len(contrasts_summary)}")

    for task, task_omnibus in omnibus_by_task.items():
        rounded_for_export(task_omnibus).to_csv(
            output_dir / f"eeg_longitudinal_omnibus_task_{task}.csv", index=False
        )
        task_contrasts = contrasts_by_task[task]
        rounded_for_export(task_contrasts).to_csv(
            output_dir / f"eeg_longitudinal_contrasts_task_{task}.csv", index=False
        )

        for family_name, measures in FEATURE_FAMILIES.items():
            for contrast_name, _, _ in CONTRASTS:
                plot_path = output_dir / f"eeg_longitudinal_plot_task_{task}_{family_name}_{contrast_name}.png"
                if len(measures) > 1:
                    make_contrast_heatmap(task_contrasts, task, family_name, measures, contrast_name, plot_path)
                else:
                    make_contrast_dotplot(task_contrasts, task, family_name, contrast_name, plot_path)

    for row in omnibus_significant.itertuples(index=False):
        plot_path = output_dir / f"eeg_longitudinal_trajectory_task_{row.Task}_{row.ROI}_{row.Measure}.png"
        make_trajectory_plot(df, row, plot_path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python eeg_longitudinal_stats_v3.py wide_feature_table.csv")
        sys.exit(1)
    main(sys.argv[1])
