"""
EDF File Inspection
============================

Run this FIRST, before any preprocessing. It scans a list of .edf files and
reports the channel names, sampling rate, duration, and event markers
found in each -- so we can build an accurate channel-name alias
dictionary and confirm marker codes for the full preprocessing pipeline.

Output
------
Prints a report per file, and saves a combined CSV
(edf_inspection_report.csv) summarizing all files at once
"""

import sys
import os
import glob
import mne
import pandas as pd

mne.set_log_level("ERROR")  # keep console output focused on our own report


def inspect_file(path):
    try:
        raw = mne.io.read_raw_edf(path, preload=False)
    except Exception as e:
        return {"file": path, "error": str(e)}

    info = raw.info
    ch_names = raw.ch_names
    sfreq = info["sfreq"]
    duration_sec = raw.n_times / sfreq

    # try to pull annotations/events (marker codes) if present
    try:
        annotations = raw.annotations
        marker_summary = {}
        for desc in annotations.description:
            marker_summary[desc] = marker_summary.get(desc, 0) + 1
    except Exception:
        marker_summary = {}

    return {
        "file": path,
        "n_channels": len(ch_names),
        "channel_names": ", ".join(ch_names),
        "sampling_rate_hz": sfreq,
        "duration_sec": round(duration_sec, 1),
        "markers_found": ", ".join(f"{k}(x{v})" for k, v in marker_summary.items()) or "NONE FOUND",
        "error": "",
    }


def main(file_paths):
    results = []
    for path in file_paths:
        print(f"\n{'=' * 70}")
        print(f"Inspecting: {path}")
        print("=" * 70)
        r = inspect_file(path)
        if r.get("error"):
            print(f"  ERROR reading file: {r['error']}")
        else:
            print(f"  Channels ({r['n_channels']}): {r['channel_names']}")
            print(f"  Sampling rate: {r['sampling_rate_hz']} Hz")
            print(f"  Duration: {r['duration_sec']} sec")
            print(f"  Markers found: {r['markers_found']}")
        results.append(r)

    df = pd.DataFrame(results)
    df.to_csv("edf_inspection_report.csv", index=False)
    print(f"\n\nSaved combined report: edf_inspection_report.csv")
    print("Send this CSV back so channel-name harmonization and marker")
    print("mapping can be built accurately for both machines.")


def resolve_file_list(args):
    """
    Accepts any mix of:
      - one or more folder paths -> each is recursively scanned for .edf/.EDF files
      - one or more explicit file paths -> used as-is
      - any combination of the above (e.g. two folders, or a folder + a file)
    """
    files = []
    for arg in args:
        if os.path.isdir(arg):
            pattern_lower = os.path.join(arg, "**", "*.edf")
            pattern_upper = os.path.join(arg, "**", "*.EDF")
            found = glob.glob(pattern_lower, recursive=True) + glob.glob(pattern_upper, recursive=True)
            print(f"Found {len(found)} .edf file(s) under: {arg}")
            files.extend(found)
        else:
            files.append(arg)
    return sorted(set(files))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python inspect_edf_files.py /path/to/folder1 /path/to/folder2 ...  (scans one or more folders)")
        print("  python inspect_edf_files.py file1.edf file2.edf ...                (or list files explicitly)")
        sys.exit(1)

    file_list = resolve_file_list(sys.argv[1:])
    if not file_list:
        print("No .edf files found. Check the folder path.")
        sys.exit(1)

    main(file_list)
