# -*- coding: utf-8 -*-
"""
aggregate_features.py
=======================

Combines all per-subject feature CSVs (produced by process_natus_subject.py
/ process_neurosoft_subject.py, one file per subject per timepoint, e.g.
"B02_pre_features.csv", "B02_post_features.csv") into ONE wide-format
table: one row per subject, columns "{task}_{feature}_pre" /
"{task}_{feature}_post" -- ready for the pre-post statistical analysis.

Usage
-----
    python aggregate_features.py

Reads every "*_features.csv" file in FEATURES_DIR, groups by subject_id
and timepoint (parsed from the filename), pivots each subject's 5-task
long-format table into a single wide row, and saves the combined result.

If you extract MORE features later (a different script, saved under a
different filename pattern, e.g. "*_features_v2.csv"), just change
FILE_PATTERN below and run again -- this produces a SEPARATE wide table
you can merge with the first one on "subject_id" (e.g. via
pandas.merge()), so you never overwrite your original feature table.
"""

import os
import re
import glob
import pandas as pd

FEATURES_DIR = "./features"
FILE_PATTERN = "*_features.csv"
OUTPUT_PATH = "./wide_feature_table.csv"

FILENAME_RE = re.compile(r"^(.+?)_(pre|post)_features\.csv$")


def parse_filename(path):
    basename = os.path.basename(path)
    m = FILENAME_RE.match(basename)
    if not m:
        print(f"  WARNING: filename doesn't match expected pattern, skipping: {basename}")
        return None, None
    return m.group(1), m.group(2)  # subject_id, timepoint


def pivot_one_file(path):
    """
    Reads one subject/timepoint's long-format CSV (rows = tasks, columns =
    features) and flattens it to a single-row wide dict:
    {"3A_Frontal_theta_relpower": value, ...}
    """
    df = pd.read_csv(path)
    row = {}
    for _, r in df.iterrows():
        task = r["task"]
        for col in df.columns:
            if col == "task":
                continue
            row[f"{task}_{col}"] = r[col]
    return row


def main():
    files = sorted(glob.glob(os.path.join(FEATURES_DIR, FILE_PATTERN)))
    print(f"Found {len(files)} feature file(s) in {FEATURES_DIR}")

    subject_rows_pre = {}
    subject_rows_post = {}

    for f in files:
        subject_id, timepoint = parse_filename(f)
        if subject_id is None:
            continue
        row = pivot_one_file(f)
        if timepoint == "pre":
            subject_rows_pre[subject_id] = row
        else:
            subject_rows_post[subject_id] = row

    subject_ids = sorted(set(subject_rows_pre.keys()) | set(subject_rows_post.keys()))
    print(f"\nFound {len(subject_ids)} subject(s):")
    for sid in subject_ids:
        has_pre = "pre" if sid in subject_rows_pre else "MISSING pre"
        has_post = "post" if sid in subject_rows_post else "MISSING post"
        flag = "  <-- INCOMPLETE" if "MISSING" in has_pre or "MISSING" in has_post else ""
        print(f"  {sid}: {has_pre}, {has_post}{flag}")

    table_rows = []
    for sid in subject_ids:
        row = {"subject_id": sid}
        pre_row = subject_rows_pre.get(sid, {})
        post_row = subject_rows_post.get(sid, {})
        all_keys = sorted(set(pre_row.keys()) | set(post_row.keys()))
        for key in all_keys:
            row[f"{key}_pre"] = pre_row.get(key)
            row[f"{key}_post"] = post_row.get(key)
        table_rows.append(row)

    wide_df = pd.DataFrame(table_rows)
    wide_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved: {OUTPUT_PATH} ({wide_df.shape[0]} subjects, {wide_df.shape[1]} columns)")


if __name__ == "__main__":
    main()
