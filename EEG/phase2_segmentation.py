"""
Phase 2 - Segmentation by Task Marker
========================================

Cuts a preprocessed continuous Raw into 60-second task blocks, using the
event markers found in edf_inspection_report.csv: "3A" (Memory encoding),
"3B" (Memory retrieval/task), "4" (Attention), "5" (Language), "6" (Math).

Handles the file-structure asymmetry between machines:
  - Natus: each file = ONE run, contains ONE occurrence of each marker
    (3A, 3B, 4, 5, 6 each appear once). segment_raw() on a single Natus
    file returns 1 segment per task.
  - Neurosoft: ONE continuous file contains all 3 runs, so each marker
    appears ~3 times. segment_raw() on a Neurosoft file returns up to 3
    segments per task directly.

Marker matching is case-insensitive (handles the "3b" vs "3B" typo seen
in B56_00) and tolerant of leading/trailing whitespace.

"Break recording" annotations (seen in some Neurosoft files) are checked
against each extracted window -- if a break falls inside a task block,
that block is flagged rather than silently kept, since the recording may
have a gap/discontinuity in it.
"""

import re
import mne

TASK_LABELS = ["3A", "3B", "4", "5", "6"]
TASK_DURATION_SEC = 60.0


def _normalize_marker(description):
    """
    Normalizes a raw annotation description to just the task label.
    Some recordings append event-comment text after the label, e.g.
    "3A(eventcomment)" or even "3A(eventcomment)(eventcomment)" -- only
    the part BEFORE the first "(" is the actual marker code.
    """
    base = description.split("(")[0]
    return base.strip().upper()


def find_task_onsets(raw):
    """
    Scans raw.annotations for task markers, returns a dict:
        {"3A": [onset_sec, ...], "3B": [...], "4": [...], "5": [...], "6": [...]}
    Onsets are in seconds relative to the start of this raw file.
    """
    onsets = {label: [] for label in TASK_LABELS}
    for annot in raw.annotations:
        desc = _normalize_marker(annot["description"])
        if desc in onsets:
            onsets[desc].append(annot["onset"])
    for label in TASK_LABELS:
        onsets[label] = sorted(onsets[label])
    return onsets


def find_break_times(raw):
    """Returns onset times (sec) of any 'Break recording' annotations."""
    breaks = []
    for annot in raw.annotations:
        if "break" in annot["description"].lower():
            breaks.append(annot["onset"])
    return breaks


def segment_raw(raw, source_label="unknown"):
    """
    Extracts a TASK_DURATION_SEC-long crop starting at each task marker
    onset found in `raw`.

    Parameters
    ----------
    raw : mne.io.Raw, already preprocessed (Phase 1 output)
    source_label : str
        Free-text tag identifying where this raw came from (e.g. filename),
        stored alongside each segment for traceability into the final
        feature table / QC logs.

    Returns
    -------
    segments : dict
        {"3A": [ {"raw": Raw, "source": str, "onset_sec": float,
                   "break_inside": bool}, ... ],
         "3B": [...], "4": [...], "5": [...], "6": [...]}
        Each task maps to a list (usually 1 entry for Natus, up to 3 for
        Neurosoft -- do not assume a fixed length).
    """
    onsets = find_task_onsets(raw)
    breaks = find_break_times(raw)
    recording_end_sec = raw.times[-1]

    segments = {label: [] for label in TASK_LABELS}

    for label, onset_list in onsets.items():
        for onset_sec in onset_list:
            tmax = onset_sec + TASK_DURATION_SEC
            if tmax > recording_end_sec:
                print(f"  WARNING [{source_label}] task {label} at {onset_sec:.1f}s "
                      f"would extend past end of recording ({recording_end_sec:.1f}s) -- skipping.")
                continue

            crop = raw.copy().crop(tmin=onset_sec, tmax=tmax, include_tmax=False)

            break_inside = any(onset_sec <= b < tmax for b in breaks)
            if break_inside:
                print(f"  WARNING [{source_label}] task {label} at {onset_sec:.1f}s "
                      f"contains a 'Break recording' event -- flagged, review before use.")

            segments[label].append({
                "raw": crop,
                "source": source_label,
                "onset_sec": onset_sec,
                "break_inside": break_inside,
            })

    # sanity check: report any task with zero occurrences found
    for label in TASK_LABELS:
        if len(segments[label]) == 0:
            print(f"  WARNING [{source_label}] no occurrences of marker '{label}' found.")

    return segments


def merge_segments_across_files(list_of_segment_dicts):
    """
    Combines segment dicts from multiple files (e.g. the 3 separate Natus
    run files for one subject/timepoint) into a single dict of
    {task_label: [all segments across all files]}.

    For Neurosoft (a single continuous file already containing all 3 runs),
    just pass a list with one element -- this function still works, it's a
    no-op merge.
    """
    merged = {label: [] for label in TASK_LABELS}
    for seg_dict in list_of_segment_dicts:
        for label in TASK_LABELS:
            merged[label].extend(seg_dict.get(label, []))
    return merged


# ----------------------------------------------------------------------
# Self-test with synthetic annotated data
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np

    print("=== Phase 2 self-test: Natus-style (1 run, 1 occurrence each) ===")
    sfreq = 500
    duration_sec = 400
    n_ch = 21
    ch_names = [f"ch{i}" for i in range(n_ch)]
    data = np.random.default_rng(0).standard_normal((n_ch, int(sfreq * duration_sec))) * 1e-5
    info = mne.create_info(ch_names, sfreq=sfreq, ch_types="eeg")
    raw_natus = mne.io.RawArray(data, info, verbose=False)

    onset_times = {"3A": 10, "3B": 80, "4": 150, "5": 220, "6": 290}
    annotations = mne.Annotations(
        onset=list(onset_times.values()),
        duration=[0] * len(onset_times),
        description=list(onset_times.keys()),
    )
    raw_natus.set_annotations(annotations)

    segs_natus = segment_raw(raw_natus, source_label="B01_01T_R1.edf")
    for label in TASK_LABELS:
        print(f"  {label}: {len(segs_natus[label])} segment(s), "
              f"duration each = {segs_natus[label][0]['raw'].times[-1]:.1f}s" if segs_natus[label] else f"  {label}: 0 segments")

    print("\n=== Phase 2 self-test: Neurosoft-style (1 file, 3 occurrences each) ===")
    duration_sec2 = 1600
    data2 = np.random.default_rng(1).standard_normal((n_ch, int(sfreq * duration_sec2))) * 1e-5
    raw_neuro = mne.io.RawArray(data2, mne.create_info(ch_names, sfreq=sfreq, ch_types="eeg"), verbose=False)

    onsets2, descs2, durs2 = [], [], []
    t = 5
    for run in range(3):
        for label in TASK_LABELS:
            onsets2.append(t)
            descs2.append(label if run != 1 else label.lower())  # simulate the "3b" lowercase typo on run 2
            durs2.append(0)
            t += 90
    raw_neuro.set_annotations(mne.Annotations(onset=onsets2, duration=durs2, description=descs2))

    segs_neuro = segment_raw(raw_neuro, source_label="B51_00.edf")
    for label in TASK_LABELS:
        print(f"  {label}: {len(segs_neuro[label])} segment(s) found (expect 3)")
        assert len(segs_neuro[label]) == 3, f"FAILED: expected 3 segments for {label}, got {len(segs_neuro[label])}"

    print("\n=== Testing merge_segments_across_files (simulating 3 Natus run files) ===")
    merged = merge_segments_across_files([segs_natus, segs_natus, segs_natus])
    for label in TASK_LABELS:
        print(f"  {label}: {len(merged[label])} segment(s) after merging 3 files (expect 3)")
        assert len(merged[label]) == 3

    print("\nPhase 2 self-test completed without errors -- case-insensitive marker")
    print("matching confirmed (caught lowercase '3b' on simulated run 2).")
