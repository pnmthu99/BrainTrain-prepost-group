"""
Phase 1 - Preprocessing
=========================

Takes a raw file (already channel-harmonized by Phase 0) through:
  1. Resample to a common sampling rate across both machines
  2. Band-pass filter + notch filter (50 Hz mains, Vietnam)
  3. Bad channel detection + interpolation
  4. Re-reference to A1 (see project discussion: the only reference scheme
     achievable on BOTH machines, since Neurosoft data is hardware-referenced
     to A1 with no access to A2 or the pre-reference signal)
  5. ICA to remove eye-blink / muscle artifacts

IMPORTANT ASSUMPTIONS (adjust the constants below once you've inspected
real data -- these are reasonable defaults, not verified against your
actual recordings yet):
  - TARGET_SFREQ = 500 Hz (Neurosoft's native rate; Natus 512->500 is a
    minor downsample, avoids upsampling Neurosoft which would fabricate
    high-frequency content that was never recorded)
  - Band-pass 0.5-45 Hz (preserves full delta band down to 1 Hz safely,
    removes slow drift below that; upper edge excludes mains harmonics)
  - ICA via extended-infomax, artifact components flagged automatically
    using frontal channels as an EOG proxy (no dedicated EOG channel in
    either machine's channel list) and MNE's muscle-artifact detector.
    Automatic ICA rejection is a starting point -- for a real study you
    should visually inspect flagged components on a subsample of files
    before trusting it on the full dataset unsupervised.
"""

import numpy as np
import mne
from phase0_channel_harmonization import harmonize_raw, CANONICAL_EEG_CHANNELS

TARGET_SFREQ = 500.0
BANDPASS_LOW = 0.5
BANDPASS_HIGH = 45.0
NOTCH_FREQ = 50.0  # Vietnam mains frequency
FLAT_THRESHOLD_UV = 1.0      # channel considered "flat"/dead below this (µV, peak-to-peak over a window)
NOISY_ZSCORE_THRESHOLD = 4.0  # channel variance z-score above this -> flagged noisy
FRONTAL_EOG_PROXY = "Fp1"     # used as an EOG-proxy channel for automatic blink detection


def set_standard_montage(raw):
    """Assigns standard 10-20 electrode positions so interpolation/ICA can use them."""
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, match_case=False, on_missing="warn")
    return raw


def detect_bad_channels_by_name(raw, channel_names):
    """
    Same flat/noisy detection as before, but restricted to an explicit list
    of channel names (used to check bads on ORIGINAL, pre-harmonization,
    pre-reference labels -- see note in preprocess_raw about why this
    must happen before re-referencing).
    """
    data = raw.get_data(picks=channel_names)
    ptp = np.ptp(data, axis=1) * 1e6
    variances = data.var(axis=1)

    flat_mask = ptp < FLAT_THRESHOLD_UV
    log_var = np.log(variances + 1e-20)
    z = (log_var - log_var.mean()) / (log_var.std() + 1e-20)
    noisy_mask = np.abs(z) > NOISY_ZSCORE_THRESHOLD

    names = np.array(channel_names)
    bads = sorted(set(names[flat_mask]) | set(names[noisy_mask]))
    return bads


def preprocess_raw(raw, machine, apply_ica=True, verbose=True):
    """
    Full Phase 1 pipeline for one raw file (one Natus run, or the full
    Neurosoft continuous recording). Handles channel harmonization AND
    re-referencing internally, in the correct order -- you do NOT need to
    call harmonize_raw() or any re-referencing function yourself first,
    just pass in the raw file exactly as loaded from disk.

    IMPORTANT ORDERING NOTE (this was a real bug caught during self-test):
    bad-channel detection MUST run on the ORIGINAL, un-referenced signal.
    If you re-reference first, a genuinely flat/dead channel (0 signal)
    becomes "0 - A1_signal" after re-referencing, which is NOT flat
    anymore -- it just looks like an inverted copy of A1, silently hiding
    a dead electrode. So the order here is: detect bads on raw labels ->
    re-reference -> harmonize names -> filter/resample -> interpolate.

    Parameters
    ----------
    raw : mne.io.Raw (already loaded, with ORIGINAL machine-specific channel
        names -- do not harmonize or re-reference before calling this)
    machine : "natus" or "neurosoft"
    apply_ica : bool
        Set False to skip ICA (e.g. for quick pipeline testing -- ICA is
        the slowest step).

    Returns
    -------
    raw : mne.io.Raw, preprocessed
    report : dict summarizing what was done (for QC logging)
    """
    report = {"machine": machine}
    raw.load_data()

    # --- Step 1: bad-channel detection on ORIGINAL labels/reference ---
    from phase0_channel_harmonization import harmonize_channel_name
    original_to_canonical = {ch: harmonize_channel_name(ch, machine) for ch in raw.ch_names}
    original_eeg_labels = [ch for ch, canon in original_to_canonical.items() if canon is not None]
    bads_original = detect_bad_channels_by_name(raw, original_eeg_labels)
    report["bad_channels_detected_pre_reref"] = bads_original

    # --- Step 2: re-reference to A1 ---
    if machine == "natus":
        raw = reref_natus_to_a1(raw)
        report["reference"] = "re-referenced to A1 (Natus)"
    elif machine == "neurosoft":
        report["reference"] = "already hardware-referenced to A1 (Neurosoft) -- no action needed"
    else:
        raise ValueError(f"Unknown machine '{machine}'")

    # --- Step 3: harmonize channel names (rename to canonical, drop non-EEG) ---
    raw, harmonize_report = harmonize_raw(raw, machine, verbose=verbose)
    report["harmonize"] = harmonize_report

    # translate the pre-reref bad-channel labels to their canonical names
    # (harmonize_raw renamed everything, so "EEG Fz-R_Fz[Fz]1" is now "Fz")
    bads_canonical = sorted({original_to_canonical[b] for b in bads_original
                              if b in original_to_canonical and original_to_canonical[b] in raw.ch_names})
    report["bad_channels_canonical"] = bads_canonical

    # --- Step 4: resample ---
    orig_sfreq = raw.info["sfreq"]
    if orig_sfreq != TARGET_SFREQ:
        raw.resample(TARGET_SFREQ)
    report["resampled_from_hz"] = orig_sfreq
    report["resampled_to_hz"] = TARGET_SFREQ

    # --- Step 5: filter ---
    raw.filter(l_freq=BANDPASS_LOW, h_freq=BANDPASS_HIGH, fir_design="firwin", verbose=verbose)
    raw.notch_filter(freqs=[NOTCH_FREQ], verbose=verbose)

    # --- Step 6: montage + interpolate the bads found back in Step 1 ---
    set_standard_montage(raw)
    raw.info["bads"] = [ch for ch in bads_canonical if ch in raw.ch_names]
    if raw.info["bads"]:
        if verbose:
            print(f"  Bad channels (detected pre-reref): {raw.info['bads']} -- interpolating.")
        raw.interpolate_bads(reset_bads=True)

    # --- Step 7: ICA ---
    if apply_ica:
        raw, ica_report = _run_ica(raw, verbose=verbose)
        report["ica"] = ica_report

    return raw, report


def _run_ica(raw, verbose=True):
    """
    Fits ICA and automatically flags likely eye-blink and muscle components.
    Uses a frontal channel as an EOG proxy since neither machine has a
    dedicated EOG channel.
    """
    ica = mne.preprocessing.ICA(n_components=0.99, method="infomax",
                                 fit_params=dict(extended=True),
                                 random_state=42, max_iter="auto")
    ica.fit(raw, verbose=verbose)

    eog_indices = []
    if FRONTAL_EOG_PROXY in raw.ch_names:
        try:
            eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=FRONTAL_EOG_PROXY, verbose=verbose)
        except Exception as e:
            if verbose:
                print(f"  EOG-proxy detection failed ({e}), skipping.")

    muscle_indices = []
    try:
        muscle_indices, muscle_scores = ica.find_bads_muscle(raw, verbose=verbose)
    except Exception as e:
        if verbose:
            print(f"  Muscle-artifact detection failed ({e}), skipping.")

    exclude = sorted(set(eog_indices) | set(muscle_indices))
    ica.exclude = exclude
    raw = ica.apply(raw, verbose=verbose)

    if verbose:
        print(f"  ICA: {len(exclude)} component(s) removed (EOG-like: {eog_indices}, muscle-like: {muscle_indices})")

    return raw, {"n_components_removed": len(exclude), "eog_like": eog_indices, "muscle_like": muscle_indices}


def reref_natus_to_a1(raw_before_harmonization):
    """
    Re-references a Natus raw file to A1. Called INTERNALLY by
    preprocess_raw() at the correct point in the pipeline (after bad-channel
    detection, before channel-name harmonization) -- you don't need to call
    this yourself.

    Must run on a raw file that still has ORIGINAL (un-harmonized) channel
    names, since it searches for the Natus-style "...R_A1[..." pattern to
    find the A1 channel.

    Neurosoft files do NOT need this -- they are already hardware-referenced
    to A1 (channel names literally end in "-A1"), so nothing to do there.
    """
    a1_candidates = [ch for ch in raw_before_harmonization.ch_names
                      if "A1" in ch.upper() and "R_A1[" in ch]
    if not a1_candidates:
        raise ValueError(
            "Could not find an A1 channel in this Natus file to re-reference to. "
            f"Available channels: {raw_before_harmonization.ch_names}"
        )
    a1_ch = a1_candidates[0]
    raw_before_harmonization.set_eeg_reference(ref_channels=[a1_ch], verbose=False)
    return raw_before_harmonization


# ----------------------------------------------------------------------
# Self-test with synthetic data (structure/API check only -- artifact
# detection thresholds should be validated against real recordings)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Phase 1 self-test (synthetic data) ===")

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
    duration_sec = 10  # short, just to test the pipeline runs end-to-end
    n_samples = int(sfreq * duration_sec)
    rng = np.random.default_rng(0)
    data = rng.standard_normal((len(natus_labels), n_samples)) * 2e-5
    # simulate one flat/dead channel
    data[5, :] = 0.0

    info = mne.create_info(natus_labels, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)

    raw_clean, report = preprocess_raw(raw, machine="natus", apply_ica=False, verbose=False)

    print(f"Final channels ({len(raw_clean.ch_names)}): {raw_clean.ch_names}")
    print(f"Final sfreq: {raw_clean.info['sfreq']}")
    print(f"Bad channels detected (pre-reref): {report['bad_channels_detected_pre_reref']}")
    print(f"Bad channels interpolated (canonical names): {report['bad_channels_canonical']}")
    assert "Fz" in report["bad_channels_canonical"], "FAILED: flat channel Fz was not caught!"
    print("Phase 1 self-test completed without errors -- flat channel correctly detected.")
