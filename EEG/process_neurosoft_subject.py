# -*- coding: utf-8 -*-
"""
process_neurosoft_subject.py
==============================

Neurosoft counterpart to process_natus_subject.py. Simpler than the Natus
version in two ways: (1) each subject/timepoint is ALREADY one continuous
file containing all 3 runs (no concatenation needed), and (2) the data is
already hardware-referenced to A1 (channel names end in "-A1"), so no
re-referencing step is needed.

Run this ONE FILE AT A TIME per subject/timepoint, same as the Natus
script -- edit CONFIG below and run.

What this script does, in order
--------------------------------
1. Load the single continuous .edf file for this subject/timepoint.
2. Harmonize channel names to the same 19-channel canonical set used for
   Natus (Fp1, Fp2, F7, F3, Fz, F4, F8, C3, Cz, C4, P3, Pz, P4, T3, T4,
   T5, T6, O1, O2) -- Neurosoft already uses old T3/T4/T5/T6 naming
   natively, so this step mainly strips the "-A1" reference suffix.
3. Filter (0.5-45 Hz + 50 Hz notch) for the main analysis signal, while
   separately branching a 1-100 Hz copy for ICA/ICLabel.
4. Build events/epochs from annotations (tmin=0, tmax=60s per task
   marker: 3A, 3B, 4, 5, 6) -- since this file contains all 3 runs, each
   marker should appear ~3 times.
5. Run ICA + ICLabel (80% threshold, eye/muscle only), remove flagged
   components.
6. PRINTS epoch counts per marker (3A/3B/4/5/6) for you to sanity-check.
7. SAVES the cleaned epochs to <subject>_<timepoint>-epo.fif (in
   EPOCHS_DIR) for later reuse without re-running ICA.
8. Extracts the agreed feature set (relative band power, theta/alpha +
   theta/beta ratio, sample entropy) and saves to
   <subject>_<timepoint>_features.csv (in FEATURES_DIR).

Usage
-----
Edit the CONFIG section below, then:
    python process_neurosoft_subject.py
"""

import os
import numpy as np
import pandas as pd
import mne
from mne.time_frequency import psd_array_welch
import antropy as ant

from phase0_channel_harmonization import (
    harmonize_raw, harmonize_channel_name, ROI_CLUSTERS, CANONICAL_EEG_CHANNELS,
)

# ======================================================================
# CONFIG -- edit this for each subject/timepoint you run
# ======================================================================
SUBJECT_ID = "B51"
TIMEPOINT = "pre"
FILE_PATH = "/mnt/data_lab513/thupnm/BrainTrain-prepost-group/EEG/raweeg_01m/B51_00.edf"

EPOCHS_DIR = "./epochs"
FEATURES_DIR = "./features"
APPLY_ICA = True

TASK_LABELS = ["3A", "3B", "4", "5", "6"]
TASK_DURATION_SEC = 60.0

TARGET_SFREQ = 500.0
BANDPASS_LOW = 0.5
BANDPASS_HIGH = 45.0
NOTCH_FREQ = 50.0

ICLABEL_THRESHOLD = 0.80
ICLABEL_AUTO_REJECT_CATEGORIES = {"eye blink", "muscle artifact"}

SUBEPOCH_SEC = 4.0
REJECT_PTP_UV = 150.0
MIN_CLEAN_SUBEPOCHS_FRACTION = 0.5
BANDS = {"delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0)}
RELATIVE_POWER_TOTAL_RANGE = (1.0, 40.0)


# ======================================================================
# Step 1-2: load, bad-channel check, harmonize (no reref needed --
# Neurosoft is already hardware-referenced to A1)
# ======================================================================
def load_and_preprocess_channels(file_path):
    print(f"\nLoading: {file_path}")
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

    # bad-channel check on ORIGINAL labels (before harmonization drops the
    # "-A1" suffix info) -- for Neurosoft there's no separate re-reference
    # step to worry about ordering against, since it's already referenced
    original_to_canonical = {ch: harmonize_channel_name(ch, "neurosoft") for ch in raw.ch_names}
    eeg_labels = [ch for ch, canon in original_to_canonical.items() if canon is not None]
    flat_original = []
    if eeg_labels:
        data = raw.get_data(picks=eeg_labels)
        ptp_uv = np.ptp(data, axis=1) * 1e6
        flat_original = [ch for ch, p in zip(eeg_labels, ptp_uv) if p < 1.0]
        if flat_original:
            print(f"  WARNING: flat/dead channel(s) detected: {flat_original}")

    raw, _ = harmonize_raw(raw, machine="neurosoft", verbose=True)

    flat_canonical = [original_to_canonical[ch] for ch in flat_original if ch in original_to_canonical]
    raw.info["bads"] = [ch for ch in flat_canonical if ch in raw.ch_names]

    return raw


# ======================================================================
# Step 3: resample, montage + interpolate, filter (+ ICA branch)
# ======================================================================
def preprocess(raw):
    orig_sfreq = raw.info["sfreq"]
    if orig_sfreq != TARGET_SFREQ:
        raw.resample(TARGET_SFREQ)

    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, match_case=False, on_missing="warn")
    if raw.info["bads"]:
        print(f"  Interpolating bad channel(s): {raw.info['bads']}")
        raw.interpolate_bads(reset_bads=True)

    raw_for_ica = None
    if APPLY_ICA:
        raw_for_ica = raw.copy().filter(l_freq=1.0, h_freq=100.0, fir_design="firwin", verbose=False)

    raw.filter(l_freq=BANDPASS_LOW, h_freq=BANDPASS_HIGH, fir_design="firwin", verbose=False)
    raw.notch_filter(freqs=[NOTCH_FREQ], verbose=False)

    return raw, raw_for_ica


# ======================================================================
# Step 4: build epochs from annotations
# ======================================================================
def _normalize_marker(description):
    return description.split("(")[0].strip().upper()


def build_epochs(raw):
    events_list = []
    for annot in raw.annotations:
        label = _normalize_marker(annot["description"])
        if label in TASK_LABELS:
            onset_sample = int(round(annot["onset"] * raw.info["sfreq"]))
            events_list.append([onset_sample, 0, TASK_LABELS.index(label) + 1])

    if not events_list:
        print("  WARNING: no task markers (3A/3B/4/5/6) found in this file's annotations!")
        print("  All annotation descriptions found:")
        for annot in raw.annotations:
            print(f"    {annot['description']!r}")
        return None

    events = np.array(sorted(events_list, key=lambda x: x[0]))
    event_id = {label: i + 1 for i, label in enumerate(TASK_LABELS)}

    print("\n  Epoch counts found per marker (expect ~3 each, one per run):")
    for label, code in event_id.items():
        n = int(np.sum(events[:, 2] == code))
        print(f"    {label}: {n}")

    recording_end_sec = raw.times[-1]
    dropped = 0
    valid_events = []
    for row in events:
        onset_sec = row[0] / raw.info["sfreq"]
        if onset_sec + TASK_DURATION_SEC > recording_end_sec:
            dropped += 1
            continue
        valid_events.append(row)
    if dropped:
        print(f"  WARNING: {dropped} marker(s) too close to end of recording, would exceed "
              f"file duration ({recording_end_sec:.1f}s) -- excluded.")

    if not valid_events:
        print("  WARNING: no valid markers remain after excluding those too close to the "
              "end of the recording -- cannot build any epochs for this file.")
        return None

    events = np.array(valid_events, dtype=int)

    # reject_by_annotation=False: MNE auto-inserts "BAD boundary" annotations
    # at concatenation seams (Natus: joining 3 run files) -- with the
    # default reject_by_annotation=True, any epoch whose 60s window happens
    # to overlap one of these seams gets SILENTLY dropped, which can lose
    # real task epochs for no data-quality reason. Our own sub-epoch
    # amplitude QC (later, in feature extraction) and ICA/ICLabel already
    # handle genuine artifact rejection, so we don't need MNE's
    # annotation-based rejection here.
    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=0, tmax=TASK_DURATION_SEC,
                         baseline=None, preload=True, reject_by_annotation=False, verbose=False)
    return epochs


# ======================================================================
# Step 5: ICA + ICLabel
# ======================================================================
def run_ica_iclabel(epochs, raw_for_ica):
    from mne_icalabel import label_components

    ica = mne.preprocessing.ICA(n_components=0.99, method="infomax",
                                 fit_params=dict(extended=True),
                                 random_state=42, max_iter="auto")

    epochs_for_ica = mne.Epochs(raw_for_ica, epochs.events, event_id=epochs.event_id,
                                 tmin=0, tmax=TASK_DURATION_SEC, baseline=None,
                                 preload=True, reject_by_annotation=False, verbose=False)
    ica.fit(epochs_for_ica, verbose=False)

    ic_labels = label_components(epochs_for_ica, ica, method="iclabel")
    labels = ic_labels["labels"]
    probs = ic_labels["y_pred_proba"]

    exclude = [
        idx for idx, (label, prob) in enumerate(zip(labels, probs))
        if label in ICLABEL_AUTO_REJECT_CATEGORIES and prob >= ICLABEL_THRESHOLD
    ]
    ica.exclude = exclude

    print(f"\n  ICA/ICLabel: {len(exclude)}/{len(labels)} component(s) removed.")
    for idx in exclude:
        print(f"    component {idx}: {labels[idx]} (p={probs[idx]:.2f})")

    epochs_clean = ica.apply(epochs.copy(), verbose=False)
    return epochs_clean, {"n_removed": len(exclude), "labels": labels, "probabilities": [round(float(p), 3) for p in probs]}


# ======================================================================
# Step 8: feature extraction (relative power, ratios, entropy) --
# identical logic to the Natus script
# ======================================================================
def make_subepochs(data, sfreq, subepoch_sec=SUBEPOCH_SEC):
    n_per = int(round(subepoch_sec * sfreq))
    n_total = data.shape[1]
    n_sub = n_total // n_per
    return [data[:, i * n_per:(i + 1) * n_per] for i in range(n_sub)]


def qc_reject_subepochs(subepochs):
    clean = []
    for sub in subepochs:
        ptp_uv = np.ptp(sub, axis=1) * 1e6
        if np.all(ptp_uv < REJECT_PTP_UV):
            clean.append(sub)
    n_total = len(subepochs)
    n_clean = len(clean)
    frac = n_clean / n_total if n_total > 0 else 0.0
    usable = frac >= MIN_CLEAN_SUBEPOCHS_FRACTION and n_clean >= 3
    return clean, {"n_total": n_total, "n_clean": n_clean, "fraction_clean": frac, "usable": usable}


def _roi_indices(ch_names, roi_channels):
    return [i for i, ch in enumerate(ch_names) if ch in roi_channels]


def compute_features_for_epoch(epoch_data, sfreq, ch_names):
    """epoch_data: (n_channels, n_samples) for ONE 60s epoch."""
    subepochs = make_subepochs(epoch_data, sfreq)
    clean, qc = qc_reject_subepochs(subepochs)

    result = {"qc": qc}
    if len(clean) == 0:
        for roi in ROI_CLUSTERS:
            for band in BANDS:
                result[f"{roi}_{band}_relpower"] = np.nan
            result[f"{roi}_theta_alpha_ratio"] = np.nan
            result[f"{roi}_theta_beta_ratio"] = np.nan
            result[f"{roi}_entropy"] = np.nan
        return result

    stacked = np.stack(clean, axis=0)
    n_fft = min(int(sfreq * 2), stacked.shape[-1])
    psds, freqs = psd_array_welch(stacked, sfreq=sfreq, fmin=RELATIVE_POWER_TOTAL_RANGE[0],
                                   fmax=RELATIVE_POWER_TOTAL_RANGE[1], n_fft=n_fft, verbose=False)
    mean_psd = psds.mean(axis=0)
    total_power = np.trapezoid(mean_psd, freqs, axis=1)

    band_power_per_channel = {}
    for band, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)
        bp = np.trapezoid(mean_psd[:, mask], freqs[mask], axis=1)
        band_power_per_channel[band] = bp / (total_power + 1e-20)

    concatenated = np.concatenate(clean, axis=1)
    entropy_per_channel = np.array([ant.sample_entropy(concatenated[i, :] * 1e6)
                                     for i in range(concatenated.shape[0])])

    for roi, roi_channels in ROI_CLUSTERS.items():
        idx = _roi_indices(ch_names, roi_channels)
        if not idx:
            for band in BANDS:
                result[f"{roi}_{band}_relpower"] = np.nan
            result[f"{roi}_theta_alpha_ratio"] = np.nan
            result[f"{roi}_theta_beta_ratio"] = np.nan
            result[f"{roi}_entropy"] = np.nan
            continue
        for band in BANDS:
            result[f"{roi}_{band}_relpower"] = float(np.mean(band_power_per_channel[band][idx]))
        theta = result[f"{roi}_theta_relpower"]
        alpha = result[f"{roi}_alpha_relpower"]
        beta = result[f"{roi}_beta_relpower"]
        result[f"{roi}_theta_alpha_ratio"] = theta / alpha if alpha else np.nan
        result[f"{roi}_theta_beta_ratio"] = theta / beta if beta else np.nan
        result[f"{roi}_entropy"] = float(np.mean(entropy_per_channel[idx]))

    return result


def extract_all_features(epochs):
    ch_names = epochs.ch_names
    sfreq = epochs.info["sfreq"]

    rows = []
    for label in TASK_LABELS:
        if label not in epochs.event_id:
            continue
        this_task_epochs = epochs[label]
        n_epochs = len(this_task_epochs)
        if n_epochs == 0:
            continue

        per_epoch_features = []
        for i in range(n_epochs):
            data = this_task_epochs.get_data(copy=True)[i]
            feats = compute_features_for_epoch(data, sfreq, ch_names)
            per_epoch_features.append(feats)

        usable = [f for f in per_epoch_features if f["qc"]["usable"]]
        print(f"  Task {label}: {len(usable)}/{n_epochs} epoch(s) usable after sub-epoch QC.")

        row = {"task": label, "n_epochs_total": n_epochs, "n_epochs_used": len(usable)}
        if usable:
            all_keys = [k for k in usable[0].keys() if k != "qc"]
            for k in all_keys:
                row[k] = float(np.nanmean([f[k] for f in usable]))
        else:
            for roi in ROI_CLUSTERS:
                for band in BANDS:
                    row[f"{roi}_{band}_relpower"] = np.nan
                row[f"{roi}_theta_alpha_ratio"] = np.nan
                row[f"{roi}_theta_beta_ratio"] = np.nan
                row[f"{roi}_entropy"] = np.nan
        rows.append(row)

    return pd.DataFrame(rows)


# ======================================================================
# Main
# ======================================================================
def main():
    print("=" * 70)
    print(f"Subject {SUBJECT_ID} ({TIMEPOINT}) -- Neurosoft")
    print("=" * 70)

    raw = load_and_preprocess_channels(FILE_PATH)
    raw, raw_for_ica = preprocess(raw)

    epochs = build_epochs(raw)
    if epochs is None:
        print("\nABORTING: no epochs could be built (see marker warning above).")
        return

    if APPLY_ICA:
        epochs, ica_report = run_ica_iclabel(epochs, raw_for_ica)

    os.makedirs(EPOCHS_DIR, exist_ok=True)
    epo_path = os.path.join(EPOCHS_DIR, f"{SUBJECT_ID}_{TIMEPOINT}-epo.fif")
    epochs.save(epo_path, overwrite=True)
    print(f"\nSaved epochs: {epo_path}")
    print(f"  (reload later with: mne.read_epochs({epo_path!r}) -- no need to redo ICA)")

    features_df = extract_all_features(epochs)
    os.makedirs(FEATURES_DIR, exist_ok=True)
    csv_path = os.path.join(FEATURES_DIR, f"{SUBJECT_ID}_{TIMEPOINT}_features.csv")
    features_df.to_csv(csv_path, index=False)
    print(f"\nSaved features: {csv_path}")
    print(features_df[["task", "n_epochs_total", "n_epochs_used"]].to_string(index=False))


if __name__ == "__main__":
    main()
