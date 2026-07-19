"""
Phase 0 - Channel Name Harmonization
=====================================

Purpose
-------
Two recording systems (Natus, Neurosoft) label channels differently, and
naming can even vary WITHIN the same machine (different technicians).
This module normalizes any raw channel label to a canonical 10-20 name,
handles the old/new temporal-electrode naming convention (T3->T7, T4->T8,
T5->P7, T6->P8), and validates that each file has the channels needed for
the chosen ROI clusters before any preprocessing happens.

Confirmed from edf_inspection_report.csv
------------------------------------------
- "Machine A" (Natus): 53 channels, 512 Hz, split into 3 files/run
  (...TR1/TR2/TR3). PSG-style system (also has ECG, Flow, Abdomen, Chest,
  SpO2, Position, PPG, CHIN, DIF, E1/E2 -- non-EEG, dropped here). Channel
  labels are truncated to EDF's 16-char limit, e.g. "EEG FP1-R_Fp1[FP" ->
  the real channel name sits between "R_" and "[". Uses MODERN temporal
  naming (T7, T8, P7, P8).
- "Machine B" (Neurosoft): 20-21 channels, 500 Hz, single continuous file
  per subject/timepoint. Already hardware-referenced to A1 (channel names
  end in "-A1"). Uses OLD temporal naming (T3, T4, T5, T6).

Both machines otherwise cover the same 19 scalp sites used in this
project's canonical channel set (matching prior BrainTrain analysis
scripts: no Fpz/Oz, temporal sites use old T3/T4/T5/T6 naming), so no
channels need to be dropped from the common set.
"""

import re
import mne

# ----------------------------------------------------------------------
# Canonical channel set (target names every raw channel gets mapped to)
# ----------------------------------------------------------------------
CANONICAL_EEG_CHANNELS = [
    "Fp1", "Fp2",
    "F7", "F3", "Fz", "F4", "F8",
    "C3", "Cz", "C4",
    "P3", "Pz", "P4",
    "T3", "T4", "T5", "T6",
    "O1", "O2",
]

# Modern (ACNS 1994) -> old (Jasper 1958) temporal electrode naming.
# This project's canonical set uses the OLD convention (T3/T4/T5/T6), to
# stay consistent with prior BrainTrain analysis scripts. Applied BEFORE
# canonical matching so files using either convention converge to T3/T4/T5/T6.
NEW_TO_OLD_TEMPORAL = {
    "T7": "T3",
    "T8": "T4",
    "P7": "T5",
    "P8": "T6",
}

# ROI clusters used for feature extraction (Phase 3). Defined on canonical
# names, so this list works identically regardless of source machine.
ROI_CLUSTERS = {
    "Frontal":  ["Fp1", "Fp2", "F3", "F4", "Fz", "F7", "F8"],
    "Central":  ["C3", "C4", "Cz"],
    "Parietal": ["P3", "P4", "Pz"],
    "Temporal": ["T3", "T4", "T5", "T6"],
    "Occipital": ["O1", "O2"],
}

# Non-EEG channels seen on the Natus/PSG-style system -- explicitly
# excluded, never mapped to a canonical EEG name.
NATUS_NON_EEG_KEYWORDS = [
    "ECG", "FLOW", "ABDOMEN", "CHEST", "SAO2", "SPO2", "POSITION",
    "PPG", "CHIN", "DIF", "E1", "E2", "RLEG",
]


def _clean_token(token):
    """Uppercase + strip whitespace for robust comparison."""
    return token.strip().upper()


def _canonical_lookup():
    """
    Builds a dict mapping UPPERCASE variants (old or new naming) to the
    canonical mixed-case name, e.g. {"T7": "T3", "T3": "T3", "FP1": "Fp1"}.
    """
    lookup = {}
    for name in CANONICAL_EEG_CHANNELS:
        lookup[_clean_token(name)] = name
    for new, old in NEW_TO_OLD_TEMPORAL.items():
        lookup[_clean_token(new)] = old
    return lookup


_CANON_LOOKUP = _canonical_lookup()


def parse_natus_label(raw_label):
    """
    Natus/PSG-style labels are truncated to 16 chars by the EDF spec, e.g.:
        "EEG FP1-R_Fp1[FP"   -> the electrode name sits between "R_" and "["
        "EEG 20-R_T7[20]"    -> "T7"
        "EEG 34-R_CHIN1[3"   -> "CHIN1" (non-EEG, will be excluded)

    Returns the extracted token (not yet canonicalized), or None if the
    expected "R_...[" pattern isn't found (e.g. truly non-standard labels).
    """
    match = re.search(r"R_([A-Za-z0-9]+)\[", raw_label)
    if match:
        return match.group(1)
    return None


def parse_neurosoft_label(raw_label):
    """
    Neurosoft labels look like "FP1-A1", "T3-A1", "FZ-A1" -- already
    hardware-referenced to A1. Strip the "-A1" (or "-A2") reference suffix
    to recover the electrode name.
    """
    return re.sub(r"-A[12]$", "", raw_label.strip(), flags=re.IGNORECASE)


def harmonize_channel_name(raw_label, machine):
    """
    Maps one raw channel label to its canonical 10-20 name.

    Parameters
    ----------
    raw_label : str
        The channel name exactly as it appears in raw.ch_names.
    machine : str
        Either "natus" or "neurosoft".

    Returns
    -------
    str or None
        Canonical name (e.g. "Fp1", "T7") if this is a recognized EEG
        scalp channel; None if it's a non-EEG channel (ECG, Flow, etc.)
        or an unrecognized label that needs manual review.
    """
    machine = machine.lower()

    if machine == "natus":
        upper_label = raw_label.upper()
        if any(kw in upper_label for kw in NATUS_NON_EEG_KEYWORDS):
            return None
        token = parse_natus_label(raw_label)
        if token is None:
            return None
    elif machine == "neurosoft":
        token = parse_neurosoft_label(raw_label)
    else:
        raise ValueError(f"Unknown machine '{machine}'. Use 'natus' or 'neurosoft'.")

    return _CANON_LOOKUP.get(_clean_token(token), None)


def harmonize_raw(raw, machine, verbose=True):
    """
    Renames all channels in an MNE Raw object to canonical 10-20 names,
    drops non-EEG / unrecognized channels, and validates that every
    channel needed for the ROI_CLUSTERS is present.

    Parameters
    ----------
    raw : mne.io.Raw
    machine : str
        "natus" or "neurosoft"
    verbose : bool
        Print a summary of the mapping applied.

    Returns
    -------
    raw : mne.io.Raw
        Modified in place AND returned, containing only canonically-named
        EEG channels.
    report : dict
        {"mapped": {...}, "dropped": [...], "missing_for_rois": [...]}
    """
    mapping = {}
    dropped = []

    for ch in raw.ch_names:
        canon = harmonize_channel_name(ch, machine)
        if canon is None:
            dropped.append(ch)
        else:
            mapping[ch] = canon

    # guard against two raw channels mapping to the same canonical name
    # (would silently collide on rename) -- flag rather than fail silently
    seen = {}
    collisions = []
    for raw_name, canon in mapping.items():
        if canon in seen:
            collisions.append((seen[canon], raw_name, canon))
        seen[canon] = raw_name
    if collisions:
        raise ValueError(
            f"Channel name collision after harmonization: {collisions}. "
            "Two different raw channels mapped to the same canonical name -- "
            "inspect this file manually before proceeding."
        )

    raw.drop_channels(dropped)
    raw.rename_channels(mapping)

    all_roi_channels = sorted({ch for roi in ROI_CLUSTERS.values() for ch in roi})
    missing = [ch for ch in all_roi_channels if ch not in raw.ch_names]

    if verbose:
        print(f"[{machine}] Mapped {len(mapping)} EEG channels, dropped {len(dropped)} non-EEG/unrecognized.")
        if dropped:
            print(f"  Dropped: {dropped}")
        if missing:
            print(f"  WARNING: missing channels needed for ROI clusters: {missing}")
        else:
            print("  All ROI-cluster channels present.")

    return raw, {"mapped": mapping, "dropped": dropped, "missing_for_rois": missing}


# ----------------------------------------------------------------------
# Self-test using the exact label patterns confirmed in
# edf_inspection_report.csv (no real data needed to verify the logic)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Testing Natus label parsing ===")
    natus_test_labels = [
        "EEG FP1-R_Fp1[FP", "EEG Fpz-R_Fpz[FP", "EEG FP2-R_Fp2[FP",
        "EEG F7-R_F7[F7]1", "EEG F3-R_F3[F3]1", "EEG Fz-R_Fz[Fz]1",
        "EEG F4-R_F4[F4]1", "EEG F8-R_F8[F8]1",
        "EEG 20-R_T7[20]", "EEG C3-R_C3[C3]1", "EEG Cz-R_Cz[Cz]1",
        "EEG C4-R_C4[C4]1", "EEG 21-R_T8[21]",
        "EEG 3-R_P7[3]", "EEG P3-R_P3[P3]1", "EEG Pz-R_Pz[Pz]1",
        "EEG P4-R_P4[P4]1", "EEG 19-R_P8[19]",
        "EEG O1-R_O1[O1]1", "EEG Oz-R_Oz[Oz]1", "EEG O2-R_O2[O2]1",
        "EEG 1-R_A1[1]", "EEG 2-R_A2[2]",
        "EEG 34-R_CHIN1[3", "ECG-LA", "SaO2", "Flow", "Abdomen", "Chest", "Position", "PPG",
    ]
    for label in natus_test_labels:
        canon = harmonize_channel_name(label, "natus")
        print(f"  {label:20s} -> {canon}")

    print("\n=== Testing Neurosoft label parsing ===")
    neurosoft_test_labels = [
        "FP1-A1", "FP2-A1", "FZ-A1", "CZ-A1", "PZ-A1", "FPZ-A1", "OZ-A1",
        "T3-A1", "T4-A1", "T5-A1", "T6-A1", "F7-A1", "F8-A1",
        "F3-A1", "F4-A1", "C3-A1", "C4-A1", "P3-A1", "P4-A1", "O1-A1", "O2-A1",
    ]
    for label in neurosoft_test_labels:
        canon = harmonize_channel_name(label, "neurosoft")
        print(f"  {label:20s} -> {canon}")

    print("\n=== Testing full harmonize_raw() on synthetic Raw objects ===")
    import numpy as np

    natus_labels_full = natus_test_labels
    info_natus = mne.create_info(natus_labels_full, sfreq=512, ch_types="eeg")
    data_natus = np.random.randn(len(natus_labels_full), 512 * 5) * 1e-5
    raw_natus = mne.io.RawArray(data_natus, info_natus, verbose=False)
    raw_natus, report_natus = harmonize_raw(raw_natus, "natus")

    print()
    neurosoft_labels_full = neurosoft_test_labels
    info_neuro = mne.create_info(neurosoft_labels_full, sfreq=500, ch_types="eeg")
    data_neuro = np.random.randn(len(neurosoft_labels_full), 500 * 5) * 1e-5
    raw_neuro = mne.io.RawArray(data_neuro, info_neuro, verbose=False)
    raw_neuro, report_neuro = harmonize_raw(raw_neuro, "neurosoft")

    print("\n=== Cross-machine channel set comparison ===")
    common = set(raw_natus.ch_names) & set(raw_neuro.ch_names)
    only_natus = set(raw_natus.ch_names) - set(raw_neuro.ch_names)
    only_neuro = set(raw_neuro.ch_names) - set(raw_natus.ch_names)
    print(f"  Common channels ({len(common)}): {sorted(common)}")
    print(f"  Only in Natus ({len(only_natus)}): {sorted(only_natus)}")
    print(f"  Only in Neurosoft ({len(only_neuro)}): {sorted(only_neuro)}")
