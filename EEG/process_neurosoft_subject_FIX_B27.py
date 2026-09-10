# -*- coding: utf-8 -*-
"""
Process one Neurosoft subject/timepoint recorded in two EDF parts.

Keep this file in the same directory as process_neurosoft_subject.py.
Edit only the CONFIG values below. FILE_PATHS must be in recording order.
"""

import os
import numpy as np
import mne

import process_neurosoft_subject as pipeline


# ======================================================================
# CONFIG -- fill these values for the interrupted recording
# ======================================================================
SUBJECT_ID = "B27"
TIMEPOINT = "pre"
FILE_PATHS = [
    "/mnt/data_lab513/thupnm/BrainTrain-prepost-group/EEG/raweeg_01m/B27_00/B27_00R1.edf",
    "/mnt/data_lab513/thupnm/BrainTrain-prepost-group/EEG/raweeg_01m/B27_00/B27_00R2R3.edf",
]

EPOCHS_DIR = "./epochs"
FEATURES_DIR = "./features"
APPLY_ICA = True


def load_and_concatenate_parts(file_paths):
    if len(file_paths) != 2:
        raise ValueError("FILE_PATHS must contain exactly two EDF files.")

    raw_parts = [pipeline.load_and_preprocess_channels(path) for path in file_paths]

    reference_channels = raw_parts[0].ch_names
    for path, raw_part in zip(file_paths[1:], raw_parts[1:]):
        if raw_part.ch_names != reference_channels:
            raise ValueError(
                "The two EDF files do not have the same harmonized EEG channels. "
                f"Check channel labels in: {path}"
            )

    # A channel marked bad in either part is interpolated after concatenation.
    all_bads = sorted({ch for raw_part in raw_parts for ch in raw_part.info["bads"]})
    for raw_part in raw_parts:
        raw_part.info["bads"] = all_bads.copy()

    first_part_duration = raw_parts[0].n_times / raw_parts[0].info["sfreq"]
    raw = mne.concatenate_raws(raw_parts, preload=True, verbose=False)

    # Add an explicit marker for the missing interval. This is preserved through
    # resampling and lets build_epochs exclude only a 60-s task that crosses the
    # power-outage seam. Epochs entirely before or after the seam are retained.
    outage_annotation = mne.Annotations(
        onset=[first_part_duration], duration=[0.0],
        description=["POWER_OUTAGE_BOUNDARY"],
        orig_time=raw.annotations.orig_time,
    )
    raw.set_annotations(raw.annotations + outage_annotation)
    return raw


def build_epochs_excluding_outage_crossings(raw):
    events_list = []
    for annot in raw.annotations:
        label = pipeline._normalize_marker(annot["description"])
        if label in pipeline.TASK_LABELS:
            onset_sample = int(round(annot["onset"] * raw.info["sfreq"]))
            events_list.append([onset_sample, 0, pipeline.TASK_LABELS.index(label) + 1])

    if not events_list:
        print("  WARNING: no task markers (3A/3B/4/5/6) found in the annotations!")
        return None

    events = np.array(sorted(events_list, key=lambda x: x[0]), dtype=int)
    all_event_id = {label: i + 1 for i, label in enumerate(pipeline.TASK_LABELS)}

    print("\n  Epoch counts found per marker (expect ~3 each, one per run):")
    for label, code in all_event_id.items():
        print(f"    {label}: {int(np.sum(events[:, 2] == code))}")

    outage_boundaries = [
        annot["onset"] for annot in raw.annotations
        if annot["description"] == "POWER_OUTAGE_BOUNDARY"
    ]
    recording_end_sec = raw.times[-1]
    dropped_end = 0
    dropped_outage = 0
    valid_events = []
    for row in events:
        onset_sec = row[0] / raw.info["sfreq"]
        if onset_sec + pipeline.TASK_DURATION_SEC > recording_end_sec:
            dropped_end += 1
            continue
        if any(onset_sec < boundary < onset_sec + pipeline.TASK_DURATION_SEC
               for boundary in outage_boundaries):
            dropped_outage += 1
            continue
        valid_events.append(row)

    if dropped_end:
        print(f"  WARNING: excluded {dropped_end} marker(s) without a full "
              f"{pipeline.TASK_DURATION_SEC:.0f}-s recording after onset.")
    if dropped_outage:
        print(f"  WARNING: excluded {dropped_outage} marker(s) whose 60-s epoch "
              "crosses the power-outage boundary.")
    if not valid_events:
        print("  WARNING: no valid task markers remain -- cannot build epochs.")
        return None

    events = np.array(valid_events, dtype=int)
    event_id = {
        label: code for label, code in all_event_id.items()
        if np.any(events[:, 2] == code)
    }
    missing_labels = [label for label in pipeline.TASK_LABELS if label not in event_id]
    if missing_labels:
        print("  WARNING: no usable epoch(s) for marker(s): "
              f"{', '.join(missing_labels)}. They will be omitted from outputs.")

    return mne.Epochs(
        raw, events, event_id=event_id, tmin=0,
        tmax=pipeline.TASK_DURATION_SEC, baseline=None, preload=True,
        reject_by_annotation=False, verbose=False,
    )


def main():
    print("=" * 70)
    print(f"Subject {SUBJECT_ID} ({TIMEPOINT}) -- Neurosoft, two EDF parts")
    print("=" * 70)

    raw = load_and_concatenate_parts(FILE_PATHS)
    raw = pipeline.preprocess(raw)
    epochs = build_epochs_excluding_outage_crossings(raw)
    if epochs is None:
        print("\nABORTING: no epochs could be built (see marker warning above).")
        return

    if APPLY_ICA:
        epochs, _ = pipeline.run_ica_iclabel(epochs)

    epochs.filter(
        l_freq=pipeline.BANDPASS_LOW, h_freq=pipeline.BANDPASS_HIGH,
        fir_design="firwin", verbose=False,
    )

    os.makedirs(EPOCHS_DIR, exist_ok=True)
    epo_path = os.path.join(EPOCHS_DIR, f"{SUBJECT_ID}_{TIMEPOINT}-epo.fif")
    epochs.save(epo_path, overwrite=True)
    print(f"\nSaved epochs: {epo_path}")

    features_df = pipeline.extract_all_features(epochs)
    os.makedirs(FEATURES_DIR, exist_ok=True)
    csv_path = os.path.join(FEATURES_DIR, f"{SUBJECT_ID}_{TIMEPOINT}_features.csv")
    features_df.to_csv(csv_path, index=False)
    print(f"\nSaved features: {csv_path}")
    print(features_df[["task", "n_epochs_total", "n_epochs_used"]].to_string(index=False))


if __name__ == "__main__":
    main()
