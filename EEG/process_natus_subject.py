# -*- coding: utf-8 -*-
"""
process_natus_subject.py
==========================

Run this ONE FILE AT A TIME per subject/timepoint (mirrors the proven
workflow of braintrain-natus-final.py, upgraded with everything decided
in the project discussion). You control the file list directly in the
CONFIG section below, run it, and inspect the printed epoch counts /
ICLabel output for that specific subject before moving to the next one.

What this script does, in order
--------------------------------
1. Load the run file(s) for ONE subject/timepoint and concatenate them
   (handles 1, 2, or 3 files -- e.g. B04 post only has R2+R3, B27 pre has
   R1 + a merged R2R3 file; just list whatever files actually exist).
2. Harmonize channel names to the 19-channel canonical set (Fp1, Fp2, F7,
   F3, Fz, F4, F8, C3, Cz, C4, P3, Pz, P4, T3, T4, T5, T6, O1, O2 -- old
   T3/T4/T5/T6 naming, no Fpz/Oz, matching prior BrainTrain scripts).
3. Re-reference to A1.
4. Filter (0.5-45 Hz + 50 Hz notch) for the main analysis signal, while
   separately branching a 1-100 Hz copy for ICA/ICLabel (see phase1 for
   why this order matters -- muscle artifact signature above 45 Hz would
   otherwise be lost before ICLabel ever sees it).
5. Build events/epochs directly from annotations (tmin=0, tmax=60s per
   task marker: 3A, 3B, 4, 5, 6) -- exactly like the proven old script,
   using mne.Epochs so overlapping/duplicate events across concatenated
   runs are handled the same way MNE always has.
6. Run ICA + ICLabel (80% threshold, eye/muscle only -- see phase1 for
   full justification), remove flagged components from the epochs.
7. PRINTS how many epochs were found per marker (3A/3B/4/5/6) -- this is
   the visibility you asked for, so you can sanity-check each file before
   trusting it.
8. SAVES the cleaned epochs to disk (<subject>_<timepoint>-epo.fif) so you
   can reload them later to extract different features WITHOUT re-running
   ICA/preprocessing. Reload with:
       epochs = mne.read_epochs("B02_pre-epo.fif")
9. Extracts the agreed feature set (relative band power, theta/alpha +
   theta/beta ratio, sample entropy -- all computed via sub-epoching
   within each 60s task epoch, averaged across epochs belonging to the
   same marker) and saves to <subject>_<timepoint>_features.csv.

Usage
-----
Edit the CONFIG section below, then:
    python process_natus_subject.py
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
SUBJECT_ID = "B02"
TIMEPOINT = "pre"
RUN_FILES = [
    "/mnt/data_lab513/thupnm/BrainTrain-prepost-group/EEG/raweeg_01m/B01_01T/B01_01TR1.edf",
    "/mnt/data_lab513/thupnm/BrainTrain-prepost-group/EEG/raweeg_01m/B01_01T/B01_01TR2.edf",
    "/mnt/data_lab513/thupnm/BrainTrain-prepost-group/EEG/raweeg_01m/B01_01T/B01_01TR3.edf",
    # add more run files here if this subject/timepoint has them, e.g.:
    # "/path/B02_00R2.edf",
    # "/path/B02_00R3.edf",
]
EPOCHS_DIR = "./epochs"      # where to save the cleaned epochs (.fif)
FEATURES_DIR = "./features"  # where to save the per-subject feature table (.csv)
APPLY_ICA = True

TASK_LABELS = ["3A", "3B", "4", "5", "6"]
TASK_DURATION_SEC = 60.0

TARGET_SFREQ = 500.0
BANDPASS_LOW = 0.5
BANDPASS_HIGH = 45.0
NOTCH_FREQ = 50.0

ICLABEL_THRESHOLD = 0.9
ICLABEL_AUTO_REJECT_CATEGORIES = {"eye blink", "muscle artifact"}

SUBEPOCH_SEC = 4.0
REJECT_PTP_UV = 150.0
MIN_CLEAN_SUBEPOCHS_FRACTION = 0.5
BANDS = {"delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0)}
RELATIVE_POWER_TOTAL_RANGE = (1.0, 40.0)


# ======================================================================
# Step 1-4: load, bad-channel check, re-reference, harmonize -- done PER
# FILE (before concatenation), then concatenate, then resample/filter/ICA-branch
# ======================================================================
def _detect_and_mark_bads(raw):
    """Bad-channel check on ORIGINAL labels/reference (before reref hides flat channels)."""
    original_to_canonical = {ch: harmonize_channel_name(ch, "natus") for ch in raw.ch_names}
    eeg_labels = [ch for ch, canon in original_to_canonical.items() if canon is not None]
    flat_original = []
    if eeg_labels:
        data = raw.get_data(picks=eeg_labels)
        ptp_uv = np.ptp(data, axis=1) * 1e6
        flat_original = [ch for ch, p in zip(eeg_labels, ptp_uv) if p < 1.0]
        if flat_original:
            print(f"  WARNING: flat/dead channel(s) detected (pre-reref): {flat_original}")
    return original_to_canonical, flat_original


def preprocess_one_run(raw):
    """Per-file steps: bad-channel check -> re-reference to A1 -> harmonize names."""
    raw.load_data()
    original_to_canonical, flat_original = _detect_and_mark_bads(raw)

    a1_candidates = [ch for ch in raw.ch_names if "R_A1[" in ch]
    if not a1_candidates:
        raise ValueError(f"No A1 channel found to re-reference to. Channels: {raw.ch_names}")
    raw.set_eeg_reference(ref_channels=[a1_candidates[0]], verbose=False)

    raw, _ = harmonize_raw(raw, machine="natus", verbose=False)

    flat_canonical = [original_to_canonical[ch] for ch in flat_original if ch in original_to_canonical]
    raw.info["bads"] = [ch for ch in flat_canonical if ch in raw.ch_names]

    return raw


def load_and_concatenate(run_files):
    print(f"\nLoading {len(run_files)} file(s):")
    raws = []
    for f in run_files:
        print(f"  {f}")
        raw = mne.io.read_raw_edf(f, preload=True, verbose=False)
        raw = preprocess_one_run(raw)  # bad-channel + reref + harmonize, per file
        raws.append(raw)

    all_bads = sorted({ch for r in raws for ch in r.info["bads"]})

    if len(raws) == 1:
        raw = raws[0]
    else:
        # all raws now share the identical canonical 19-channel structure,
        # so concatenation is safe regardless of original per-file naming
        raw = mne.concatenate_raws(raws)
    raw.info["bads"] = all_bads  # keep the union of bads found across all runs

    return raw


# ======================================================================
# Step 5 (was 2-4): resample, montage + interpolate, filter (+ ICA branch)
# -- done AFTER concatenation, on the now-uniform canonical-channel raw
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
# Step 5: build epochs from annotations
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

    events = np.array(sorted(events_list, key=lambda x: x[0]), dtype=int)
    event_id = {label: i + 1 for i, label in enumerate(TASK_LABELS)}

    print("\n  Epoch counts found per marker:")
    for label, code in event_id.items():
        n = int(np.sum(events[:, 2] == code))
        print(f"    {label}: {n}")

    # guard against markers too close to the end of a (possibly truncated --
    # some Natus files in this dataset were confirmed not properly closed)
    # recording, which would make a 60s epoch window run past available data
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
              f"file duration ({recording_end_sec:.1f}s) -- excluded. This can happen with "
              f"truncated/improperly-closed recordings -- double check this file if unexpected.")

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
# Step 6: ICA + ICLabel
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
# Step 9: feature extraction (relative power, ratios, entropy)
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
            data = this_task_epochs.get_data(copy=True)[i]  # (n_channels, n_samples)
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
    print(f"Subject {SUBJECT_ID} ({TIMEPOINT}) -- Natus")
    print("=" * 70)

    raw = load_and_concatenate(RUN_FILES)
    raw, raw_for_ica = preprocess(raw)

    epochs = build_epochs(raw)
    if epochs is None:
        print("\nABORTING: no epochs could be built (see marker warning above).")
        return

    if APPLY_ICA:
        epochs, ica_report = run_ica_iclabel(epochs, raw_for_ica)

    epo_path = os.path.join(EPOCHS_DIR, f"{SUBJECT_ID}_{TIMEPOINT}-epo.fif")
    os.makedirs(EPOCHS_DIR, exist_ok=True)
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
