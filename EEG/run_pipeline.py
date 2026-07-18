"""
run_pipeline.py - Full EEG Pipeline Orchestrator
====================================================

Ties Phases 0-4 together into one command. Handles the file-structure
difference between machines:
  - Natus: 3 separate .edf files per subject/timepoint (one per run)
  - Neurosoft: 1 continuous .edf file per subject/timepoint (contains all
    3 runs already)

Usage
-----
Edit SUBJECT_FILE_MAP below (or build it programmatically from your
folder structure) to point at your real .edf files, then run:

    python run_pipeline.py

This will produce eeg_feature_table.csv -- one row per subject, columns
"{task}_{ROI}_{feature}_pre" / "..._post" -- ready to feed into the same
kind of pre-post statistical analysis already built for the cognitive
tests (paired t-test / Wilcoxon + FDR, adapted for a much larger number
of columns here).

IMPORTANT: this file has NOT been run against real .edf data yet (no
real files were available in this environment). Each phase has been
validated individually against synthetic data matching the exact channel
label / marker patterns confirmed from your edf_inspection_report.csv.
Run this on 1-2 real subjects first and check the printed QC output
carefully before trusting it on the full dataset.
"""

import mne
import pandas as pd
from phase1_preprocessing import preprocess_raw
from phase2_segmentation import segment_raw, merge_segments_across_files, TASK_LABELS
from phase4_aggregate import build_subject_timepoint_row, build_wide_table
from phase0_channel_harmonization import CANONICAL_EEG_CHANNELS

# ----------------------------------------------------------------------
# EDIT THIS: map each subject to their machine and their pre/post file(s).
# Natus subjects: list of 3 files (one per run) for each timepoint.
# Neurosoft subjects: list of exactly 1 file for each timepoint.
# ----------------------------------------------------------------------
from discovered_subject_file_map import SUBJECT_FILE_MAP
SUBJECT_FILE_MAP = {"B02": SUBJECT_FILE_MAP["B02"]}

APPLY_ICA = False       # set False for a quick structural test run (much faster, skips artifact removal)
OUTPUT_CSV = "eeg_feature_table.csv"
QC_LOG_CSV = "eeg_qc_log.csv"


def process_one_file(file_path, machine, apply_ica=APPLY_ICA):
    """
    Loads one .edf file, runs Phase 1 (preprocessing) and Phase 2
    (segmentation), returns the segments dict for that single file.
    """
    print(f"\n--- Loading {file_path} ({machine}) ---")
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

    raw_clean, prep_report = preprocess_raw(raw, machine=machine, apply_ica=apply_ica, verbose=False)
    print(f"  Preprocessing done. Bad channels interpolated: {prep_report['bad_channels_canonical']}")

    segments = segment_raw(raw_clean, source_label=file_path)
    return segments, prep_report


def process_subject_timepoint(file_list, machine, apply_ica=APPLY_ICA):
    """
    Handles the Natus (3 files) vs Neurosoft (1 file) asymmetry: loads
    every file in file_list, segments each, merges into one
    {task_label: [all segments across all files]} dict.
    """
    all_segment_dicts = []
    all_prep_reports = []
    for file_path in file_list:
        segments, prep_report = process_one_file(file_path, machine, apply_ica)
        all_segment_dicts.append(segments)
        all_prep_reports.append(prep_report)

    merged = merge_segments_across_files(all_segment_dicts)
    return merged, all_prep_reports


def run_full_pipeline(subject_file_map=SUBJECT_FILE_MAP, apply_ica=APPLY_ICA,
                       output_csv=OUTPUT_CSV, qc_log_csv=QC_LOG_CSV):
    """
    Runs the full pipeline for every subject in subject_file_map, saves
    the final wide-format feature table and a QC log.
    """
    if not subject_file_map:
        raise ValueError(
            "SUBJECT_FILE_MAP is empty. Edit run_pipeline.py to point at your "
            "real .edf files before running the full pipeline."
        )

    subject_rows_pre = {}
    subject_rows_post = {}
    qc_log_rows = []

    for subject_id, info in subject_file_map.items():
        machine = info["machine"]
        print(f"\n{'=' * 70}\nSubject {subject_id} ({machine})\n{'=' * 70}")

        print(f"\n[PRE] Processing {len(info['pre_files'])} file(s)...")
        merged_pre, prep_reports_pre = process_subject_timepoint(info["pre_files"], machine, apply_ica)
        row_pre, meta_pre = build_subject_timepoint_row(merged_pre, sfreq=500.0, ch_names=CANONICAL_EEG_CHANNELS)
        subject_rows_pre[subject_id] = row_pre

        print(f"\n[POST] Processing {len(info['post_files'])} file(s)...")
        merged_post, prep_reports_post = process_subject_timepoint(info["post_files"], machine, apply_ica)
        row_post, meta_post = build_subject_timepoint_row(merged_post, sfreq=500.0, ch_names=CANONICAL_EEG_CHANNELS)
        subject_rows_post[subject_id] = row_post

        for timepoint, meta in [("pre", meta_pre), ("post", meta_post)]:
            for task, m in meta.items():
                qc_log_rows.append({
                    "subject_id": subject_id,
                    "timepoint": timepoint,
                    "task": task,
                    "n_runs_total": m["n_runs_total"],
                    "n_runs_used": m["n_runs_used"],
                })

    wide_table = build_wide_table(subject_rows_pre, subject_rows_post)
    wide_table.to_csv(output_csv, index=False)
    print(f"\n\nSaved feature table: {output_csv} ({wide_table.shape[0]} subjects, {wide_table.shape[1]} columns)")

    qc_log = pd.DataFrame(qc_log_rows)
    qc_log.to_csv(qc_log_csv, index=False)
    print(f"Saved QC log: {qc_log_csv}")
    print("\nCheck qc_log.csv for any task with n_runs_used < 2 -- those subject/task/timepoint")
    print("combinations rest on very little clean data and may need manual review.")

    return wide_table, qc_log


# ----------------------------------------------------------------------
# Self-test: verifies the full Phase 0-4 wiring with synthetic raw data,
# WITHOUT reading any real .edf files (bypasses read_raw_edf, injects
# synthetic Raw objects directly into the same preprocess/segment/aggregate
# call chain used by the real pipeline above).
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        print("=== run_pipeline.py end-to-end self-test (synthetic data, no real files) ===\n")
        import numpy as np

        def make_synthetic_natus_raw(seed):
            natus_labels = [
                "EEG FP1-R_Fp1[FP", "EEG Fpz-R_Fpz[FP", "EEG FP2-R_Fp2[FP",
                "EEG F7-R_F7[F7]1", "EEG F3-R_F3[F3]1", "EEG Fz-R_Fz[Fz]1",
                "EEG F4-R_F4[F4]1", "EEG F8-R_F8[F8]1",
                "EEG 20-R_T7[20]", "EEG C3-R_C3[C3]1", "EEG Cz-R_Cz[Cz]1",
                "EEG C4-R_C4[C4]1", "EEG 21-R_T8[21]",
                "EEG 3-R_P7[3]", "EEG P3-R_P3[P3]1", "EEG Pz-R_Pz[Pz]1",
                "EEG P4-R_P4[P4]1", "EEG 19-R_P8[19]",
                "EEG O1-R_O1[O1]1", "EEG Oz-R_Oz[Oz]1", "EEG O2-R_O2[O2]1",
                "EEG 1-R_A1[1]", "EEG 2-R_A2[2]",
            ]
            sfreq = 512
            duration = 400
            rng = np.random.default_rng(seed)
            data = rng.standard_normal((len(natus_labels), int(sfreq * duration))) * 1e-5
            info = mne.create_info(natus_labels, sfreq=sfreq, ch_types="eeg")
            raw = mne.io.RawArray(data, info, verbose=False)
            onset_times = {"3A": 10, "3B": 90, "4": 170, "5": 250, "6": 330}
            raw.set_annotations(mne.Annotations(
                onset=list(onset_times.values()), duration=[0] * 5,
                description=list(onset_times.keys())))
            return raw

        # monkey-patch mne.io.read_raw_edf for this self-test only, so
        # process_one_file() can be exercised without real files on disk
        _original_read = mne.io.read_raw_edf
        _fake_files = {}

        def _fake_read_raw_edf(file_path, preload=True, verbose=False):
            return _fake_files[file_path]

        mne.io.read_raw_edf = _fake_read_raw_edf

        fake_map = {}
        for i, sid in enumerate(["S001"]):
            pre_files = [f"{sid}_pre_R{r}.edf" for r in range(1, 2)]   # 1 run only, for a fast wiring check
            post_files = [f"{sid}_post_R{r}.edf" for r in range(1, 2)]
            for j, f in enumerate(pre_files):
                _fake_files[f] = make_synthetic_natus_raw(seed=i * 10 + j)
            for j, f in enumerate(post_files):
                _fake_files[f] = make_synthetic_natus_raw(seed=i * 10 + j + 100)
            fake_map[sid] = {"machine": "natus", "pre_files": pre_files, "post_files": post_files}

        wide_table, qc_log = run_full_pipeline(
            subject_file_map=fake_map, apply_ica=False,
            output_csv="selftest_feature_table.csv", qc_log_csv="selftest_qc_log.csv",
        )

        mne.io.read_raw_edf = _original_read  # restore

        assert wide_table.shape[0] == 1, "FAILED: expected 1 subject in output table"
        assert (qc_log["n_runs_used"] == 1).all(), "FAILED: expected the single clean run used (no artifacts injected)"
        print("\n\nSELF-TEST PASSED (reduced scope: 1 subject, 1 run/timepoint) -- "
              "full Phase 0->4 pipeline wiring confirmed working end-to-end.")
        print("NOTE: entropy computation is the slow step at full scale (3 runs x 5 tasks x 21 channels")
        print("per subject/timepoint) -- see chat for performance discussion before running the full dataset.")
    else:
        run_full_pipeline()
