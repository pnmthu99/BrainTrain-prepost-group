# -*- coding: utf-8 -*-
"""
EEG longitudinal analysis with an omnibus-then-post-hoc workflow.

This script requires eeg_longitudinal_stats_v3.py in the same folder. It first
applies FDR to Friedman omnibus tests within each task x feature family. Only
features with omnibus q < .05 receive three Wilcoxon post-hoc contrasts;
their three p-values are adjusted with Holm's method within that feature.

All outputs are written to results_posthoc/ in the current working directory.

Usage:
    python eeg_longitudinal_omnibus_posthoc_stats.py wide_feature_table.csv
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eeg_longitudinal_stats_v3 import (
    ALPHA,
    CONTRASTS,
    FEATURE_FAMILIES,
    ROIS,
    TIMEPOINTS,
    analyze_feature,
    fdr_bh,
    make_trajectory_plot,
    sig_stars,
)


def holm_adjust(p_values):
    """Return Holm-adjusted p-values for one feature's planned contrasts."""
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p_values), np.nan)
    valid = ~np.isnan(p_values)
    if not valid.any():
        return adjusted

    values = p_values[valid]
    order = np.argsort(values)
    ranked = values[order]
    n_values = len(values)
    corrected = (n_values - np.arange(n_values)) * ranked
    corrected = np.maximum.accumulate(corrected)
    values_adjusted = np.empty(n_values)
    values_adjusted[order] = np.clip(corrected, 0, 1)
    adjusted[valid] = values_adjusted
    return adjusted


def add_omnibus_fdr(results):
    results = results.copy()
    results["p_adj_FDR"] = fdr_bh(results["p_value"].to_numpy())
    results["Significant_after_FDR"] = results["p_adj_FDR"] < ALPHA
    return results


def run_all_tasks(df):
    omnibus_by_task = {}
    posthoc_by_task = {}
    visualization_by_task = {}

    for task in ["3B", "4", "5", "6"]:
        task_omnibus = []
        task_posthoc = []
        task_visualization = []
        for family_name, measures in FEATURE_FAMILIES.items():
            family_omnibus = []
            family_contrasts = []

            for roi in ROIS:
                for measure in measures:
                    base = f"{task}_{roi}_{measure}"
                    columns = {timepoint: f"{base}_{timepoint}" for timepoint in TIMEPOINTS}
                    if not all(column in df.columns for column in columns.values()):
                        continue

                    omnibus, contrasts = analyze_feature(df, columns)
                    if omnibus is None:
                        continue
                    omnibus.update({
                        "Task": task,
                        "Family": family_name,
                        "ROI": roi,
                        "Measure": measure,
                    })
                    family_omnibus.append(omnibus)
                    for contrast in contrasts:
                        contrast.update({
                            "Task": task,
                            "Family": family_name,
                            "ROI": roi,
                            "Measure": measure,
                        })
                        family_contrasts.append(contrast)

            if not family_omnibus:
                continue

            omnibus_family = add_omnibus_fdr(pd.DataFrame(family_omnibus))
            task_omnibus.append(omnibus_family)
            contrast_family = pd.DataFrame(family_contrasts).merge(
                omnibus_family[["Task", "Family", "ROI", "Measure", "p_adj_FDR"]],
                on=["Task", "Family", "ROI", "Measure"],
                how="left",
            ).rename(columns={"p_adj_FDR": "Omnibus_p_adj_FDR"})
            contrast_family["Passed_omnibus_FDR"] = contrast_family["Omnibus_p_adj_FDR"] < ALPHA
            contrast_family["p_adj_Holm"] = np.nan
            contrast_family["Significant_after_Holm"] = False

            for omnibus_row in omnibus_family.itertuples(index=False):
                if not omnibus_row.Significant_after_FDR:
                    continue

                selected = (
                    (contrast_family["ROI"] == omnibus_row.ROI)
                    & (contrast_family["Measure"] == omnibus_row.Measure)
                )
                adjusted = holm_adjust(contrast_family.loc[selected, "p_value"].to_numpy())
                contrast_family.loc[selected, "p_adj_Holm"] = adjusted
                contrast_family.loc[selected, "Significant_after_Holm"] = adjusted < ALPHA
                task_posthoc.append(contrast_family.loc[selected].copy())

            task_visualization.append(contrast_family)

        if task_omnibus:
            omnibus_by_task[task] = pd.concat(task_omnibus, ignore_index=True)
            if task_posthoc:
                posthoc_by_task[task] = pd.concat(task_posthoc, ignore_index=True)
            else:
                posthoc_by_task[task] = pd.DataFrame()
            visualization_by_task[task] = pd.concat(task_visualization, ignore_index=True)
        else:
            print(f"WARNING: no valid Pre/Post/3m features found for task {task}")

    return omnibus_by_task, posthoc_by_task, visualization_by_task


def make_posthoc_heatmap(task_df, task, family_name, measures, contrast_name, out_path):
    subset = task_df[
        (task_df["Family"] == family_name) & (task_df["Contrast"] == contrast_name)
    ]
    if subset.empty:
        return False

    effects = subset.pivot(index="ROI", columns="Measure", values="Effect_Size")
    adjusted_p = subset.pivot(index="ROI", columns="Measure", values="p_adj_Holm")
    effects = effects.reindex(index=ROIS, columns=measures)
    adjusted_p = adjusted_p.reindex(index=ROIS, columns=measures)
    values = effects.to_numpy(dtype=float)
    maximum = np.nanmax(np.abs(values)) if not np.all(np.isnan(values)) else 1.0

    fig, ax = plt.subplots(figsize=(max(5.5, 1.8 * len(measures)), 5))
    image = ax.imshow(values, cmap="RdBu_r", vmin=-maximum, vmax=maximum, aspect="auto")
    ax.set_xticks(range(len(measures)))
    ax.set_xticklabels(measures, rotation=45, ha="right")
    ax.set_yticks(range(len(ROIS)))
    ax.set_yticklabels(ROIS)
    for row in range(len(ROIS)):
        for column in range(len(measures)):
            adjusted = adjusted_p.to_numpy()[row, column]
            if pd.notna(adjusted) and adjusted < ALPHA:
                ax.text(column, row, f"{sig_stars(adjusted)}\np_Holm={adjusted:.3f}",
                        ha="center", va="center", color="white", fontweight="bold", fontsize=9)

    ax.set_title(
        f"Task {task} - {family_name} - {contrast_name}\n"
        "All effects shown; text = omnibus-FDR and Holm-significant post-hoc contrast"
    )
    fig.colorbar(image, ax=ax, label="Rank-biserial r", shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def make_posthoc_dotplot(task_df, task, family_name, contrast_name, out_path):
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
        significant = bool(row["Significant_after_Holm"])
        color = "#C1272D" if row["Mean_Diff"] > 0 else "#1F77B4"
        ax.errorbar(
            row["Mean_Diff"], position,
            xerr=[[row["Mean_Diff"] - row["Boot_95CI_Low"]],
                  [row["Boot_95CI_High"] - row["Mean_Diff"]]],
            fmt="o", color=color, ecolor=color, capsize=4, markersize=8,
            markerfacecolor=color if significant else "white", markeredgecolor=color,
        )
        if significant:
            ax.text(row["Boot_95CI_High"], position,
                    f"  {sig_stars(row['p_adj_Holm'])} p_Holm={row['p_adj_Holm']:.3f}",
                    va="center", fontsize=9)

    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_yticks(range(len(ROIS)))
    ax.set_yticklabels(ROIS)
    ax.set_xlabel("Mean difference with 95% bootstrap CI")
    ax.set_title(
        f"Task {task} - {family_name} - {contrast_name}\n"
        "Filled marker = omnibus-FDR and Holm-significant post-hoc contrast"
    )
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
    output_dir = Path("results_posthoc")
    output_dir.mkdir(exist_ok=True)
    print(f"Loaded {csv_path}: {len(df)} subjects, {len(df.columns)} columns")

    omnibus_by_task, posthoc_by_task, visualization_by_task = run_all_tasks(df)
    if not omnibus_by_task:
        raise ValueError("No features with valid Pre, Post, and 3m columns were found.")

    omnibus_summary = pd.concat(omnibus_by_task.values(), ignore_index=True)
    posthoc_tables = [table for table in posthoc_by_task.values() if not table.empty]
    posthoc_summary = pd.concat(posthoc_tables, ignore_index=True) if posthoc_tables else pd.DataFrame()
    visualization_summary = pd.concat(visualization_by_task.values(), ignore_index=True)
    rounded_for_export(omnibus_summary).to_csv(output_dir / "eeg_omnibus_summary.csv", index=False)
    rounded_for_export(posthoc_summary).to_csv(output_dir / "eeg_posthoc_summary.csv", index=False)
    rounded_for_export(visualization_summary).to_csv(
        output_dir / "eeg_all_contrasts_for_visualization.csv", index=False
    )
    print(f"Saved omnibus summary: {output_dir / 'eeg_omnibus_summary.csv'}")
    print(f"Saved post-hoc summary: {output_dir / 'eeg_posthoc_summary.csv'}")

    for task, omnibus_table in omnibus_by_task.items():
        rounded_for_export(omnibus_table).to_csv(output_dir / f"eeg_omnibus_task_{task}.csv", index=False)
        posthoc_table = posthoc_by_task[task]
        if not posthoc_table.empty:
            rounded_for_export(posthoc_table).to_csv(output_dir / f"eeg_posthoc_task_{task}.csv", index=False)
        visualization_table = visualization_by_task[task]
        rounded_for_export(visualization_table).to_csv(
            output_dir / f"eeg_all_contrasts_for_visualization_task_{task}.csv", index=False
        )

        for family_name, measures in FEATURE_FAMILIES.items():
            for contrast_name, _, _ in CONTRASTS:
                plot_path = output_dir / f"eeg_posthoc_task_{task}_{family_name}_{contrast_name}.png"
                if len(measures) > 1:
                    make_posthoc_heatmap(visualization_table, task, family_name, measures, contrast_name, plot_path)
                else:
                    make_posthoc_dotplot(visualization_table, task, family_name, contrast_name, plot_path)

    for row in omnibus_summary[omnibus_summary["Significant_after_FDR"]].itertuples(index=False):
        plot_path = output_dir / f"eeg_trajectory_task_{row.Task}_{row.ROI}_{row.Measure}.png"
        make_trajectory_plot(df, row, plot_path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python eeg_longitudinal_omnibus_posthoc_stats.py wide_feature_table.csv")
        sys.exit(1)
    main(sys.argv[1])
