"""
Phase 3 - Feature Extraction
==============================

Takes one 60-second task segment (Phase 2 output) and extracts the
confirmed feature set:
  1. Relative band power (delta, theta, alpha, beta) per ROI cluster
  2. Theta/Alpha and Theta/Beta ratio per ROI cluster
  3. Sample Entropy (broadband) per ROI cluster

Follows the sub-epoching principle established in the BrainTrain EEG study
(the preprocessing-error lesson learned there): PSD/entropy are computed
on short sub-epochs and averaged, NOT on the raw continuous 60s block.

Sub-epoch QC: sub-epochs exceeding a peak-to-peak amplitude threshold are
rejected individually (not the whole 60s block). A run is only flagged as
unusable if too few clean sub-epochs remain for a stable estimate -- that
decision happens in Phase 4 (aggregation across the 3 runs), using the
QC metadata returned here.
"""

import numpy as np
import antropy as ant
from mne.time_frequency import psd_array_welch
from phase0_channel_harmonization import ROI_CLUSTERS

SUBEPOCH_SEC = 4.0
REJECT_PTP_UV = 150.0          # peak-to-peak amplitude threshold for sub-epoch rejection
MIN_CLEAN_SUBEPOCHS_FRACTION = 0.5  # flag the run as unreliable below this fraction clean

BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}
RELATIVE_POWER_TOTAL_RANGE = (1.0, 40.0)  # denominator range for relative power


def make_subepochs(data, sfreq, subepoch_sec=SUBEPOCH_SEC):
    """
    Splits a (n_channels, n_samples) array into non-overlapping sub-epochs
    of subepoch_sec each. Drops any trailing partial sub-epoch.

    Returns
    -------
    subepochs : list of (n_channels, n_subepoch_samples) arrays
    """
    n_samples_per_subepoch = int(round(subepoch_sec * sfreq))
    n_total = data.shape[1]
    n_subepochs = n_total // n_samples_per_subepoch

    subepochs = []
    for i in range(n_subepochs):
        start = i * n_samples_per_subepoch
        end = start + n_samples_per_subepoch
        subepochs.append(data[:, start:end])
    return subepochs


def qc_reject_subepochs(subepochs):
    """
    Rejects sub-epochs where ANY channel exceeds REJECT_PTP_UV peak-to-peak.
    (Simple, transparent, inspectable criterion -- not a black-box method.)

    Returns
    -------
    clean_subepochs : list of arrays that passed QC
    qc_report : dict with counts, for Phase 4's run-level inclusion decision
    """
    clean = []
    for sub in subepochs:
        ptp_uv = np.ptp(sub, axis=1) * 1e6
        if np.all(ptp_uv < REJECT_PTP_UV):
            clean.append(sub)

    n_total = len(subepochs)
    n_clean = len(clean)
    fraction_clean = n_clean / n_total if n_total > 0 else 0.0

    qc_report = {
        "n_total_subepochs": n_total,
        "n_clean_subepochs": n_clean,
        "fraction_clean": fraction_clean,
        "usable": fraction_clean >= MIN_CLEAN_SUBEPOCHS_FRACTION and n_clean >= 3,
    }
    return clean, qc_report


def _roi_channel_indices(ch_names, roi_channels):
    """Returns the indices in ch_names that belong to this ROI (case-sensitive canonical names)."""
    return [i for i, ch in enumerate(ch_names) if ch in roi_channels]


def compute_band_power_features(clean_subepochs, sfreq, ch_names):
    """
    Computes relative band power per ROI cluster, averaged across all
    clean sub-epochs.

    Returns
    -------
    dict: {roi_name: {band_name: relative_power_value, ...}, ...}
    """
    if len(clean_subepochs) == 0:
        return {roi: {band: np.nan for band in BANDS} for roi in ROI_CLUSTERS}

    # stack sub-epochs into one array for a single psd_array_welch call:
    # shape (n_subepochs, n_channels, n_samples)
    stacked = np.stack(clean_subepochs, axis=0)
    n_samples = stacked.shape[-1]
    n_fft = min(int(sfreq * 2), n_samples)  # 2-second windows within Welch, capped by sub-epoch length

    psds, freqs = psd_array_welch(
        stacked, sfreq=sfreq, fmin=RELATIVE_POWER_TOTAL_RANGE[0], fmax=RELATIVE_POWER_TOTAL_RANGE[1],
        n_fft=n_fft, verbose=False,
    )
    # psds shape: (n_subepochs, n_channels, n_freqs) -> average across sub-epochs
    mean_psd = psds.mean(axis=0)  # (n_channels, n_freqs)

    total_power = np.trapezoid(mean_psd, freqs, axis=1)  # (n_channels,)

    band_power_per_channel = {}
    for band_name, (lo, hi) in BANDS.items():
        band_mask = (freqs >= lo) & (freqs < hi)
        band_power = np.trapezoid(mean_psd[:, band_mask], freqs[band_mask], axis=1)
        band_power_per_channel[band_name] = band_power / (total_power + 1e-20)

    result = {}
    for roi_name, roi_channels in ROI_CLUSTERS.items():
        idx = _roi_channel_indices(ch_names, roi_channels)
        if not idx:
            result[roi_name] = {band: np.nan for band in BANDS}
            continue
        result[roi_name] = {
            band: float(np.mean(band_power_per_channel[band][idx]))
            for band in BANDS
        }
    return result


def compute_ratio_features(band_power_result):
    """
    Given the output of compute_band_power_features(), derives
    theta/alpha and theta/beta ratios per ROI.
    """
    ratios = {}
    for roi_name, bands in band_power_result.items():
        theta = bands["theta"]
        alpha = bands["alpha"]
        beta = bands["beta"]
        ratios[roi_name] = {
            "theta_alpha_ratio": theta / alpha if alpha and not np.isnan(alpha) and alpha != 0 else np.nan,
            "theta_beta_ratio": theta / beta if beta and not np.isnan(beta) and beta != 0 else np.nan,
        }
    return ratios


def compute_entropy_features(clean_subepochs, ch_names):
    """
    Computes Sample Entropy per ROI cluster. Concatenates all clean
    sub-epochs per channel first (more stable estimate than averaging many
    tiny 4s-window entropy values), then averages across channels within
    each ROI.
    """
    if len(clean_subepochs) == 0:
        return {roi: np.nan for roi in ROI_CLUSTERS}

    # concatenate along time axis: (n_channels, n_subepochs * n_samples)
    concatenated = np.concatenate(clean_subepochs, axis=1)

    entropy_per_channel = np.array([
        ant.sample_entropy(concatenated[i, :] * 1e6)  # scale to µV, sample_entropy is scale-sensitive via its tolerance r
        for i in range(concatenated.shape[0])
    ])

    result = {}
    for roi_name, roi_channels in ROI_CLUSTERS.items():
        idx = _roi_channel_indices(ch_names, roi_channels)
        if not idx:
            result[roi_name] = np.nan
            continue
        result[roi_name] = float(np.mean(entropy_per_channel[idx]))
    return result


def extract_features_for_segment(segment, sfreq, ch_names):
    """
    Full Phase 3 pipeline for ONE task segment (one run's 60s block).

    Parameters
    ----------
    segment : dict, one element from Phase 2's segments[task_label] list
        (has "raw", "source", "onset_sec", "break_inside")
    sfreq : float
    ch_names : list of str (canonical ROI-cluster-relevant channel names)

    Returns
    -------
    features : dict
        {"band_power": {...}, "ratios": {...}, "entropy": {...}}
    qc : dict
        Sub-epoch QC report, used by Phase 4 to decide whether to include
        this run in the cross-run average.
    """
    data = segment["raw"].get_data(picks=ch_names)
    subepochs = make_subepochs(data, sfreq)
    clean_subepochs, qc_report = qc_reject_subepochs(subepochs)

    qc_report["break_inside"] = segment["break_inside"]
    qc_report["source"] = segment["source"]

    band_power = compute_band_power_features(clean_subepochs, sfreq, ch_names)
    ratios = compute_ratio_features(band_power)
    entropy = compute_entropy_features(clean_subepochs, ch_names)

    features = {"band_power": band_power, "ratios": ratios, "entropy": entropy}
    return features, qc_report


# ----------------------------------------------------------------------
# Self-test: synthetic signal with a KNOWN dominant frequency, verify the
# relative power pipeline correctly identifies it
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import mne

    print("=== Phase 3 self-test: synthetic alpha-dominant signal ===")
    sfreq = 500
    duration_sec = 60
    ch_names = ["Fp1", "Fp2", "F3", "F4", "Fz", "F7", "F8",  # Frontal
                "C3", "C4", "Cz",                              # Central
                "P3", "P4", "Pz",                               # Parietal
                "T7", "T8", "P7", "P8",                         # Temporal
                "O1", "O2", "Oz"]                                # Occipital
    n_ch = len(ch_names)
    t = np.arange(int(sfreq * duration_sec)) / sfreq
    rng = np.random.default_rng(0)

    # Occipital channels get a strong 10 Hz (alpha) signal + noise;
    # everything else is pure noise. Relative alpha power should come out
    # clearly higher in Occipital ROI than in, say, Central ROI.
    data = rng.standard_normal((n_ch, len(t))) * 1e-6
    alpha_signal = np.sin(2 * np.pi * 10 * t) * 3e-5
    for i, ch in enumerate(ch_names):
        if ch in ROI_CLUSTERS["Occipital"]:
            data[i, :] += alpha_signal

    info = mne.create_info(ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    fake_segment = {"raw": raw, "source": "synthetic_test", "onset_sec": 0.0, "break_inside": False}

    features, qc = extract_features_for_segment(fake_segment, sfreq, ch_names)

    print(f"QC: {qc['n_clean_subepochs']}/{qc['n_total_subepochs']} sub-epochs clean, usable={qc['usable']}")
    print("\nRelative alpha power by ROI (expect Occipital >> others):")
    for roi in ROI_CLUSTERS:
        print(f"  {roi:12s}: {features['band_power'][roi]['alpha']:.4f}")

    occipital_alpha = features["band_power"]["Occipital"]["alpha"]
    central_alpha = features["band_power"]["Central"]["alpha"]
    assert occipital_alpha > central_alpha * 2, (
        f"FAILED: expected Occipital alpha ({occipital_alpha:.4f}) to be clearly higher "
        f"than Central alpha ({central_alpha:.4f})"
    )
    print("\nPASSED: synthetic alpha signal correctly localized to Occipital ROI.")

    print("\nSample entropy by ROI:")
    for roi in ROI_CLUSTERS:
        print(f"  {roi:12s}: {features['entropy'][roi]:.4f}")

    print("\nTheta/Alpha and Theta/Beta ratios by ROI:")
    for roi in ROI_CLUSTERS:
        r = features["ratios"][roi]
        print(f"  {roi:12s}: theta/alpha={r['theta_alpha_ratio']:.4f}, theta/beta={r['theta_beta_ratio']:.4f}")

    print("\nPhase 3 self-test completed without errors.")
