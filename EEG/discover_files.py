"""
discover_files.py - Auto-discover subjects/timepoints/runs from a folder
============================================================================

Scans a folder (recursively, so it finds files inside subfolders too --
matches the Natus pattern of one subfolder per recording) and automatically
groups .edf files into subjects, timepoints (pre/post), and runs, then
detects which machine each file came from by actually peeking at its
channel count / sampling rate (NOT by guessing from filename/subject ID
ranges, which would be fragile).

CRITICAL: this does NOT run the pipeline. It only DISCOVERS and REPORTS
the grouping it found, saved to discovered_files_report.csv, so you can
visually verify the grouping is correct before feeding it into
run_pipeline.py. Misgrouped runs would silently corrupt the whole
downstream analysis, so this check is not optional.

Filename convention assumed (adjust FILENAME_PATTERN / TIMEPOINT_MAP below
if your actual files don't match):
  - Subject ID: a "B" followed by digits, e.g. "B01", "B51"
  - Timepoint: "_00" = pre/baseline, "_01T" = post (case-insensitive)
  - Natus run files: an "R1"/"R2"/"R3" (or "TR1"/"TR2"/"TR3") suffix
    somewhere in the filename indicates which of the 3 runs it is
  - Everything else (no run suffix found) is treated as a single
    continuous file (matches Neurosoft)

Usage
-----
    python discover_files.py /path/to/raweeg_01m
"""

import os
import re
import sys
import glob
import mne
import pandas as pd

mne.set_log_level("ERROR")

SUBJECT_ID_PATTERN = re.compile(r"(B\d+)", re.IGNORECASE)
TIMEPOINT_PATTERNS = {
    "pre": re.compile(r"_00(?!\d)", re.IGNORECASE),   # "_00" not followed by another digit
    "post": re.compile(r"_01T", re.IGNORECASE),
}
RUN_PATTERN = re.compile(r"T?R(\d)", re.IGNORECASE)  # matches "R1","R2","R3","TR1","TR2","TR3"

# Machine signatures confirmed from edf_inspection_report.csv
MACHINE_SIGNATURES = {
    "natus":     {"n_channels_min": 40, "sfreq": 512},
    "neurosoft": {"n_channels_max": 25, "sfreq": 500},
}


TASK_MARKERS = {"3A", "3B", "4", "5", "6"}


def check_task_markers(file_path):
    """
    Quick check (annotations only, no data load) of which task markers
    are actually present in this file. Flags files with FEW or ZERO task
    markers -- this can catch cases where a "baseline" file turns out to
    be a different kind of recording (e.g. resting-state only) rather
    than the same task battery as post, which would silently break
    Phase 2 segmentation later (0 segments found, all-NaN features).
    """
    try:
        raw = mne.io.read_raw_edf(file_path, preload=False)
    except Exception:
        return set()
    found = set()
    for annot in raw.annotations:
        # strip trailing "(eventcomment)" suffixes some recordings append,
        # e.g. "3A(eventcomment)" -- only text before the first "(" is the
        # actual marker code (see phase2_segmentation._normalize_marker)
        desc = annot["description"].split("(")[0].strip().upper()
        if desc in TASK_MARKERS:
            found.add(desc)
    return found


def detect_machine(file_path):
    """Peeks at a file's header only (no data load) to detect the machine."""
    try:
        raw = mne.io.read_raw_edf(file_path, preload=False)
    except Exception as e:
        return None, f"ERROR reading file: {e}"

    n_ch = len(raw.ch_names)
    sfreq = raw.info["sfreq"]

    if n_ch >= MACHINE_SIGNATURES["natus"]["n_channels_min"] and sfreq == MACHINE_SIGNATURES["natus"]["sfreq"]:
        return "natus", None
    elif n_ch <= MACHINE_SIGNATURES["neurosoft"]["n_channels_max"] and sfreq == MACHINE_SIGNATURES["neurosoft"]["sfreq"]:
        return "neurosoft", None
    else:
        return None, f"UNRECOGNIZED signature: {n_ch} channels @ {sfreq} Hz (doesn't match either known machine)"


def parse_filename(file_path):
    """Extracts subject_id, timepoint, run_number from a file path."""
    basename = os.path.basename(file_path)

    subject_match = SUBJECT_ID_PATTERN.search(basename)
    subject_id = subject_match.group(1).upper() if subject_match else None

    timepoint = None
    for tp, pattern in TIMEPOINT_PATTERNS.items():
        if pattern.search(basename):
            timepoint = tp
            break

    run_match = RUN_PATTERN.search(basename)
    run_number = int(run_match.group(1)) if run_match else None

    return subject_id, timepoint, run_number


def discover(folder):
    edf_files = sorted(set(
        glob.glob(os.path.join(folder, "**", "*.edf"), recursive=True) +
        glob.glob(os.path.join(folder, "**", "*.EDF"), recursive=True)
    ))
    print(f"Found {len(edf_files)} .edf file(s) under {folder}\n")

    records = []
    for file_path in edf_files:
        subject_id, timepoint, run_number = parse_filename(file_path)
        machine, error = detect_machine(file_path)
        markers_found = check_task_markers(file_path)

        records.append({
            "file": file_path,
            "subject_id": subject_id,
            "timepoint": timepoint,
            "run_number": run_number,
            "machine": machine,
            "n_task_markers_found": len(markers_found),
            "task_markers_found": ",".join(sorted(markers_found)) if markers_found else "NONE",
            "issue": error or "",
        })

    df = pd.DataFrame(records)

    # flag anything that failed to parse cleanly -- print prominently
    problems = df[df["subject_id"].isna() | df["timepoint"].isna() | df["machine"].isna()]
    if len(problems) > 0:
        print(f"!!! {len(problems)} file(s) could not be fully parsed -- REVIEW THESE MANUALLY !!!\n")
        print(problems.to_string(index=False))
        print()

    marker_problems = df[df["n_task_markers_found"] < 5]
    if len(marker_problems) > 0:
        print(f"!!! {len(marker_problems)} file(s) are missing one or more task markers "
              f"(expected all 5: 3A,3B,4,5,6) -- REVIEW THESE, may not be task recordings !!!\n")
        print(marker_problems[["file", "timepoint", "n_task_markers_found", "task_markers_found"]].to_string(index=False))
        print()

    df.to_csv("discovered_files_report.csv", index=False)
    print(f"Saved full report: discovered_files_report.csv\n")

    return df


def build_subject_file_map(df):
    """
    Converts the discovered file table into the SUBJECT_FILE_MAP format
    used by run_pipeline.py. Groups Natus run files together, sorted by
    run_number; Neurosoft (no run_number) stays as a single-file list.
    """
    clean_df = df.dropna(subset=["subject_id", "timepoint", "machine"])

    subject_map = {}
    for subject_id, sub_df in clean_df.groupby("subject_id"):
        machines = sub_df["machine"].unique()
        if len(machines) > 1:
            print(f"WARNING: subject {subject_id} has files from multiple machines "
                  f"({machines}) -- skipping, needs manual review.")
            continue
        machine = machines[0]

        entry = {"machine": machine}
        for timepoint in ["pre", "post"]:
            tp_files = sub_df[sub_df["timepoint"] == timepoint].copy()
            if len(tp_files) == 0:
                entry[f"{timepoint}_files"] = []
                continue
            # sort by run_number (NaN/None run_number sorts first -- fine for
            # single-file Neurosoft where run_number is always None)
            tp_files = tp_files.sort_values("run_number", na_position="first")
            entry[f"{timepoint}_files"] = tp_files["file"].tolist()

        subject_map[subject_id] = entry

    return subject_map


def print_summary(subject_map, df):
    print("=" * 70)
    print("DISCOVERED SUBJECT FILE MAP -- REVIEW BEFORE RUNNING THE PIPELINE")
    print("=" * 70)
    for subject_id, entry in subject_map.items():
        n_pre = len(entry["pre_files"])
        n_post = len(entry["post_files"])

        # cross-check against the marker report rather than assuming a
        # fixed run count -- pre and post may legitimately have different
        # file structures (e.g. baseline as 1 file, post split into 3 runs)
        sub_files = entry["pre_files"] + entry["post_files"]
        marker_counts = df[df["file"].isin(sub_files)]["n_task_markers_found"]
        flag = ""
        if (marker_counts < 5).any():
            flag = "  <-- one or more files missing task markers, SEE marker warning above"

        print(f"  {subject_id} ({entry['machine']}): {n_pre} pre file(s), {n_post} post file(s){flag}")
    print()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python discover_files.py /path/to/raweeg_01m")
        sys.exit(1)

    folder = sys.argv[1]
    df = discover(folder)
    subject_map = build_subject_file_map(df)
    print_summary(subject_map, df)

    # save as a Python-importable file so run_pipeline.py can use it directly
    with open("discovered_subject_file_map.py", "w") as f:
        f.write("# Auto-generated by discover_files.py -- REVIEW BEFORE USE\n")
        f.write("# Import this into run_pipeline.py: \n")
        f.write("#   from discovered_subject_file_map import SUBJECT_FILE_MAP\n\n")
        f.write("SUBJECT_FILE_MAP = {\n")
        for subject_id, entry in subject_map.items():
            f.write(f"    {subject_id!r}: {{\n")
            f.write(f"        'machine': {entry['machine']!r},\n")
            f.write(f"        'pre_files': {entry['pre_files']!r},\n")
            f.write(f"        'post_files': {entry['post_files']!r},\n")
            f.write(f"    }},\n")
        f.write("}\n")

    print("Saved: discovered_subject_file_map.py")
    print("\nNEXT STEP: open discovered_files_report.csv AND discovered_subject_file_map.py,")
    print("verify the grouping looks correct, THEN in run_pipeline.py replace:")
    print("    SUBJECT_FILE_MAP = { ... }")
    print("with:")
    print("    from discovered_subject_file_map import SUBJECT_FILE_MAP")
