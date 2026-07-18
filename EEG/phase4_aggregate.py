"""
Phase 4 - Aggregate Across Runs Into a Feature Table
========================================================

Takes the merged segments for one subject/timepoint (Phase 2's
merge_segments_across_files output: {task_label: [segment, segment, segment]})
and:
  1. Extracts features per run (Phase 3)
  2. Includes/excludes each run based on its sub-epoch QC (a run is
     excluded from the average only if too few clean sub-epochs remain --
     see phase3's MIN_CLEAN_SUBEPOCHS_FRACTION -- NOT simply averaged in
     regardless of quality)
  3. Averages the included runs into one value per subject/task/feature
  4. Flattens everything into the wide-format table used throughout this
     project (same convention as the cognitive-test analysis scripts):
     one row per subject, columns named "{task}_{ROI}_{feature}_{pre|post}"

Memory (3A, 3B) are treated as fully separate tasks throughout -- this
matches the earlier decision to FDR-correct them as two independent
families downstream in the statistics step.
"""

import numpy as np
import pandas as pd
from phase2_segmentation import TASK_LABELS
from phase3_feature_extraction import extract_features_for_segment
from phase0_channel_harmonization import ROI_CLUSTERS, CANONICAL_EEG_CHANNELS

BAND_NAMES = ["delta", "theta", "alpha", "beta"]
ROI_NAMES = list(ROI_CLUSTERS.keys())


def aggregate_task_features(segments_list, sfreq, ch_names=None, task_label=""):
    """
    Extracts and averages features across all usable runs for ONE task.

    Parameters
    ----------
    segments_list : list of segment dicts (Phase 2 output, already merged
        across files for this subject/timepoint/task)
    sfreq : float
    ch_names : list of str, defaults to CANONICAL_EEG_CHANNELS
    task_label : str, only used for warning messages

    Returns
    -------
    flat_features : dict
        {"{ROI}_delta_relpower": value, ..., "{ROI}_theta_alpha_ratio": value,
         ..., "{ROI}_entropy": value, ...}
        NaN for any ROI/feature if zero runs were usable.
    meta : dict
        {"n_runs_total": int, "n_runs_used": int, "run_qc": [qc_dict, ...]}
    """
    if ch_names is None:
        ch_names = CANONICAL_EEG_CHANNELS

    all_features = []
    all_qc = []
    for segment in segments_list:
        features, qc = extract_features_for_segment(segment, sfreq, ch_names)
        all_features.append(features)
        all_qc.append(qc)

    usable_features = [f for f, qc in zip(all_features, all_qc) if qc["usable"]]
    n_total = len(segments_list)
    n_used = len(usable_features)

    if n_used == 0:
        print(f"  WARNING [{task_label}] 0/{n_total} runs usable -- all features will be NaN for this task.")

    flat_features = {}
    for roi in ROI_NAMES:
        for band in BAND_NAMES:
            key = f"{roi}_{band}_relpower"
            if n_used == 0:
                flat_features[key] = np.nan
            else:
                vals = [f["band_power"][roi][band] for f in usable_features]
                flat_features[key] = float(np.nanmean(vals))

        for ratio_name in ["theta_alpha_ratio", "theta_beta_ratio"]:
            key = f"{roi}_{ratio_name}"
            if n_used == 0:
                flat_features[key] = np.nan
            else:
                vals = [f["ratios"][roi][ratio_name] for f in usable_features]
                flat_features[key] = float(np.nanmean(vals))

        key = f"{roi}_entropy"
        if n_used == 0:
            flat_features[key] = np.nan
        else:
            vals = [f["entropy"][roi] for f in usable_features]
            flat_features[key] = float(np.nanmean(vals))

    meta = {"n_runs_total": n_total, "n_runs_used": n_used, "run_qc": all_qc}
    return flat_features, meta


def build_subject_timepoint_row(merged_segments_by_task, sfreq, ch_names=None):
    """
    For ONE subject at ONE timepoint: runs aggregate_task_features() for
    every task (3A, 3B, 4, 5, 6) and combines into one flat dict with
    task-prefixed keys, e.g. "3A_Frontal_theta_relpower".

    Parameters
    ----------
    merged_segments_by_task : dict
        {"3A": [...], "3B": [...], "4": [...], "5": [...], "6": [...]}
        (output of phase2.merge_segments_across_files)

    Returns
    -------
    row : dict, flat, ready to become one row (for this timepoint) in the
        wide-format table
    meta_all_tasks : dict, {task_label: meta_dict} for QC logging
    """
    row = {}
    meta_all_tasks = {}
    for task_label in TASK_LABELS:
        segments_list = merged_segments_by_task.get(task_label, [])
        flat_features, meta = aggregate_task_features(segments_list, sfreq, ch_names, task_label=task_label)
        meta_all_tasks[task_label] = meta
        for feature_key, value in flat_features.items():
            row[f"{task_label}_{feature_key}"] = value
    return row, meta_all_tasks


def build_wide_table(subject_rows_pre, subject_rows_post):
    """
    Combines pre and post per-subject rows into the final wide-format
    feature table (same convention as the cognitive-test scripts:
    subject_id + "{...}_pre" / "{...}_post" columns).

    Parameters
    ----------
    subject_rows_pre : dict {subject_id: row_dict}   (row_dict from
        build_subject_timepoint_row, pre-intervention)
    subject_rows_post : dict {subject_id: row_dict}  (post-intervention)

    Returns
    -------
    pd.DataFrame, one row per subject
    """
    subject_ids = sorted(set(subject_rows_pre.keys()) | set(subject_rows_post.keys()))
    table_rows = []
    for sid in subject_ids:
        row = {"subject_id": sid}
        pre_row = subject_rows_pre.get(sid, {})
        post_row = subject_rows_post.get(sid, {})

        all_feature_keys = sorted(set(pre_row.keys()) | set(post_row.keys()))
        for key in all_feature_keys:
            row[f"{key}_pre"] = pre_row.get(key, np.nan)
            row[f"{key}_post"] = post_row.get(key, np.nan)
        table_rows.append(row)

    return pd.DataFrame(table_rows)


# ----------------------------------------------------------------------
# Self-test: 3 synthetic runs per task, one run deliberately very noisy
# (should be excluded from the average by QC)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import mne
    from phase2_segmentation import TASK_LABELS

    print("=== Phase 4 self-test: 3 runs, 1 deliberately noisy (should be excluded) ===")
    sfreq = 500
    duration_sec = 60
    ch_names = CANONICAL_EEG_CHANNELS
    n_ch = len(ch_names)

    def make_fake_segment(seed, noisy=False, source="run"):
        rng = np.random.default_rng(seed)
        t = np.arange(int(sfreq * duration_sec)) / sfreq
        data = rng.standard_normal((n_ch, len(t))) * 1e-6
        data += np.sin(2 * np.pi * 6 * t) * 1e-5  # shared theta component
        if noisy:
            # inject huge artifact spikes throughout -> every sub-epoch should fail QC
            spike_positions = rng.integers(0, len(t), size=50)
            data[:, spike_positions] += 500e-6
        info = mne.create_info(ch_names, sfreq=sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, info, verbose=False)
        return {"raw": raw, "source": source, "onset_sec": 0.0, "break_inside": False}

    merged_segments_by_task = {}
    for task in TASK_LABELS:
        merged_segments_by_task[task] = [
            make_fake_segment(seed=1, noisy=False, source=f"{task}_run1"),
            make_fake_segment(seed=2, noisy=False, source=f"{task}_run2"),
            make_fake_segment(seed=3, noisy=True, source=f"{task}_run3_BAD"),
        ]

    row, meta = build_subject_timepoint_row(merged_segments_by_task, sfreq, ch_names)

    for task in TASK_LABELS:
        m = meta[task]
        print(f"  Task {task}: {m['n_runs_used']}/{m['n_runs_total']} runs used "
              f"(expect 2/3 -- the noisy run should be excluded)")
        assert m["n_runs_used"] == 2, f"FAILED: expected 2 usable runs for task {task}, got {m['n_runs_used']}"

    print(f"\nFlat row has {len(row)} feature columns. Sample values:")
    for k in list(row.keys())[:5]:
        print(f"  {k}: {row[k]:.4f}")

    print("\n=== Testing build_wide_table with 2 fake subjects (pre/post) ===")
    fake_pre = {"S001": row, "S002": row}
    fake_post = {"S001": row, "S002": row}
    wide_df = build_wide_table(fake_pre, fake_post)
    print(f"Wide table shape: {wide_df.shape} (rows=subjects, cols=subject_id + features x 2 timepoints)")
    print(f"Sample columns: {list(wide_df.columns[:6])}")
    assert wide_df.shape[0] == 2
    assert "subject_id" in wide_df.columns

    print("\nPhase 4 self-test completed without errors -- QC-based run exclusion confirmed working.")
