# -*- coding: utf-8 -*-
"""Aggregate and analyze EEG relative power with longitudinal LMMs.

The input is a folder containing source files named
{subject}_{pre|post|3m}_features.csv. Each source file contains task rows and
ROI/band feature columns. The script aggregates relative power directly from
these source files, checks that the four bands sum to approximately one, then
fits one LMM per task x ROI x band using the planned-contrast strategy. FDR
covers 20 bands/ROIs for omnibus tests and 60 feature x contrast tests within
each task.

Requires eeg_longitudinal_lmm_stats.py in the same folder.
Usage: python eeg_relative_power_lmm.py features_folder
"""

import re
import sys
from pathlib import Path

import pandas as pd

from eeg_longitudinal_lmm_stats import (
    ALPHA,
    ROIS,
    TASKS,
    TIMEPOINTS,
    analyze_feature,
    fdr_bh,
    get_subject_ids,
    make_lmm_heatmap,
    make_lmm_trajectory,
    rounded_for_export,
)


BANDS = ["delta", "theta", "alpha", "beta"]
FAMILY = "relative_power"
SOURCE_FILE_PATTERN = re.compile(r"^(?P<subject>.+)_(?P<timepoint>pre|post|3m)_features\.csv$", re.IGNORECASE)


def source_task_column(columns):
    for column in columns:
        if column.strip().lower() == "task":
            return column
    raise ValueError("Source file has no task column.")


def load_source_features(features_dir):
    features_dir = Path(features_dir)
    if not features_dir.is_dir():
        raise ValueError(f"Not a directory: {features_dir}")

    subject_records = {}
    manifest_rows = []
    source_files = sorted(features_dir.glob("*_features.csv"))
    if not source_files:
        raise ValueError(f"No *_features.csv files found in {features_dir}")

    for source_file in source_files:
        match = SOURCE_FILE_PATTERN.match(source_file.name)
        if match is None:
            manifest_rows.append({"Source_File": source_file.name, "Status": "skipped: unexpected filename"})
            continue
        subject = match.group("subject")
        timepoint = match.group("timepoint").lower()
        source = pd.read_csv(source_file)
        task_column = source_task_column(source.columns)
        record = subject_records.setdefault(subject, {"subject_id": subject})

        for _, row in source.iterrows():
            task = str(row[task_column]).strip()
            if task not in TASKS:
                continue
            epoch_used_column = next((column for column in source.columns if column.strip() == "n_epochs_used"), None)
            epoch_used = row[epoch_used_column] if epoch_used_column else pd.NA
            manifest_rows.append({
                "Source_File": source_file.name,
                "subject_id": subject,
                "timepoint": timepoint,
                "Task": task,
                "n_epochs_used": epoch_used,
                "Status": "loaded",
            })
            for roi in ROIS:
                for band in BANDS:
                    source_column = f"{roi}_{band}_relpower"
                    output_column = f"{task}_{roi}_{band}_relpower_{timepoint}"
                    if source_column not in source.columns:
                        raise ValueError(f"{source_file.name} is missing column {source_column}")
                    if output_column in record:
                        raise ValueError(f"Duplicate source value for {subject}, {output_column}")
                    record[output_column] = pd.to_numeric(row[source_column], errors="coerce")

    wide = pd.DataFrame(subject_records.values()).sort_values("subject_id").reset_index(drop=True)
    return wide, pd.DataFrame(manifest_rows)


def build_relative_power_long(df, subject_ids):
    frames = []
    for task in TASKS:
        for roi in ROIS:
            for band in BANDS:
                base = f"{task}_{roi}_{band}_relpower"
                columns = {timepoint: f"{base}_{timepoint}" for timepoint in TIMEPOINTS}
                if not all(column in df.columns for column in columns.values()):
                    continue
                wide = pd.DataFrame({"subject_id": subject_ids})
                for timepoint, column in columns.items():
                    wide[timepoint] = df[column]
                long = wide.melt(id_vars="subject_id", value_vars=TIMEPOINTS,
                                 var_name="timepoint", value_name="relative_power")
                long["Task"] = task
                long["ROI"] = roi
                long["Band"] = band
                frames.append(long)
    if not frames:
        raise ValueError("No relative-power columns were found.")
    return pd.concat(frames, ignore_index=True)[
        ["subject_id", "Task", "ROI", "Band", "timepoint", "relative_power"]
    ]


def build_qc(long):
    complete_bands = long.pivot_table(
        index=["subject_id", "Task", "ROI", "timepoint"], columns="Band",
        values="relative_power", aggfunc="first", observed=False
    ).reindex(columns=BANDS)
    qc = complete_bands.copy()
    qc["N_bands_present"] = qc.notna().sum(axis=1)
    qc["Relative_Power_Sum"] = qc[BANDS].sum(axis=1, min_count=1)
    qc["Sum_Deviation_From_One"] = qc["Relative_Power_Sum"] - 1
    qc["All_Bands_Present"] = qc["N_bands_present"] == len(BANDS)
    qc["Sum_Within_0_001"] = qc["All_Bands_Present"] & qc["Sum_Deviation_From_One"].abs().le(0.001)
    return qc.reset_index()


def run_models(df, subject_ids):
    omnibus_by_task, contrasts_by_task, trajectories, failures = {}, {}, [], []
    for task in TASKS:
        omnibus_rows, contrast_rows = [], []
        for roi in ROIS:
            for band in BANDS:
                measure = f"{band}_relpower"
                base = f"{task}_{roi}_{measure}"
                columns = {timepoint: f"{base}_{timepoint}" for timepoint in TIMEPOINTS}
                if not all(column in df.columns for column in columns.values()):
                    failures.append({"Task": task, "ROI": roi, "Band": band,
                                     "Reason": "missing one or more required columns"})
                    continue
                result, status = analyze_feature(df, subject_ids, columns)
                if result is None:
                    failures.append({"Task": task, "ROI": roi, "Band": band, "Reason": status})
                    continue
                omnibus, contrasts, long, model_means = result
                metadata = {"Task": task, "Family": FAMILY, "ROI": roi, "Measure": measure, "Band": band}
                omnibus.update(metadata)
                omnibus_rows.append(omnibus)
                for contrast in contrasts:
                    contrast.update(metadata)
                    contrast_rows.append(contrast)
                trajectories.append((metadata, long, model_means))

        if not omnibus_rows:
            continue
        omnibus = pd.DataFrame(omnibus_rows)
        omnibus["p_adj_FDR"] = fdr_bh(omnibus["p_value"].to_numpy())
        omnibus["Significant_after_FDR"] = omnibus["p_adj_FDR"] < ALPHA
        contrasts = pd.DataFrame(contrast_rows)
        contrasts["p_adj_FDR"] = fdr_bh(contrasts["p_value"].to_numpy())
        contrasts["Significant_after_FDR"] = contrasts["p_adj_FDR"] < ALPHA
        omnibus_by_task[task] = omnibus
        contrasts_by_task[task] = contrasts
    return omnibus_by_task, contrasts_by_task, trajectories, pd.DataFrame(failures)


def main(features_dir):
    df, source_manifest = load_source_features(features_dir)
    subject_ids, subject_id_source = get_subject_ids(df)
    output_dir = Path("results_relative_power_lmm")
    output_dir.mkdir(exist_ok=True)
    print(f"Loaded {len(source_manifest)} task rows from {features_dir}: {len(df)} subjects; subject ID: {subject_id_source}")

    long = build_relative_power_long(df, subject_ids)
    qc = build_qc(long)
    omnibus_by_task, contrasts_by_task, trajectories, failures = run_models(df, subject_ids)
    if not omnibus_by_task:
        raise ValueError("No relative-power LMMs could be fitted.")

    omnibus_summary = pd.concat(omnibus_by_task.values(), ignore_index=True)
    contrasts_summary = pd.concat(contrasts_by_task.values(), ignore_index=True)
    omnibus_summary["Analysis_Strategy"] = "planned_contrasts"
    contrasts_summary["Analysis_Strategy"] = "planned_contrasts"
    df.to_csv(output_dir / "relative_power_wide_from_source.csv", index=False)
    source_manifest.to_csv(output_dir / "relative_power_source_manifest.csv", index=False)
    long.to_csv(output_dir / "relative_power_long.csv", index=False)
    rounded_for_export(qc).to_csv(output_dir / "relative_power_qc.csv", index=False)
    rounded_for_export(omnibus_summary).to_csv(output_dir / "relative_power_lmm_omnibus_summary.csv", index=False)
    rounded_for_export(contrasts_summary).to_csv(output_dir / "relative_power_lmm_contrasts_summary.csv", index=False)
    failures.to_csv(output_dir / "relative_power_lmm_failures.csv", index=False)

    q_by_feature = omnibus_summary.set_index(["Task", "ROI", "Measure"])["p_adj_FDR"]
    for task, task_omnibus in omnibus_by_task.items():
        rounded_for_export(task_omnibus).to_csv(output_dir / f"relative_power_lmm_omnibus_task_{task}.csv", index=False)
        task_contrasts = contrasts_by_task[task]
        rounded_for_export(task_contrasts).to_csv(output_dir / f"relative_power_lmm_contrasts_task_{task}.csv", index=False)
        for contrast_name in ["post_minus_pre", "3m_minus_pre", "3m_minus_post"]:
            make_lmm_heatmap(task_contrasts, task, FAMILY, [f"{band}_relpower" for band in BANDS],
                             contrast_name, output_dir / f"relative_power_task_{task}_{contrast_name}.png")

    for metadata, feature_long, model_means in trajectories:
        key = (metadata["Task"], metadata["ROI"], metadata["Measure"])
        make_lmm_trajectory(
            metadata, feature_long, model_means, q_by_feature.loc[key],
            output_dir / f"relative_power_trajectory_task_{key[0]}_{key[1]}_{metadata['Band']}.png"
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python eeg_relative_power_lmm.py features_folder")
        sys.exit(1)
    main(sys.argv[1])
