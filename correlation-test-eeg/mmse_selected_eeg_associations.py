# -*- coding: utf-8 -*-
"""Feature-informed MMSE associations for EEG cells with longitudinal effects.

For each supplied EEG family, this script reads its long data, omnibus LMM
summary and planned-contrast summary.  It then:

1. selects task x ROI x measure cells with omnibus EEG q-FDR < .05;
2. fits MMSE ~ timepoint + EEG_within + EEG_between + (1 | subject) for every
   selected cell;
3. applies BH-FDR once across all selected EEG_within coefficients, regardless
   of EEG family;
4. runs a secondary 0m-to-3m Spearman analysis only for cells with a
   q-FDR-significant 3m-minus-0m EEG contrast, with a separate FDR correction.

This is a feature-informed secondary analysis because the candidate EEG cells
were selected from time effects in the same dataset. It does not establish a
causal brain-behaviour mechanism.

Usage
-----
    python3 mmse_selected_eeg_associations.py cogtest_013m.csv \
      --eeg-family results_entropy_lmm/entropy_long.csv \
                   results_entropy_lmm/entropy_lmm_omnibus_summary.csv \
                   results_entropy_lmm/entropy_lmm_contrasts_summary.csv \
      --eeg-family results_relative_power_lmm/relative_power_long.csv \
                   results_relative_power_lmm/relative_power_lmm_omnibus_summary.csv \
                   results_relative_power_lmm/relative_power_lmm_contrasts_summary.csv
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

try:
    import matplotlib.pyplot as plt
except ImportError as error:
    raise SystemExit("This script requires matplotlib: pip install matplotlib") from error

try:
    import statsmodels.api as sm
except ImportError as error:
    raise SystemExit("This script requires statsmodels: pip install statsmodels") from error


ALPHA = 0.05
TIMEPOINTS = ["pre", "post", "3m"]
CELL_KEYS = ["Task", "ROI", "Measure"]


def fdr_bh(p_values):
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p_values), np.nan)
    valid = np.isfinite(p_values)
    if not valid.any():
        return adjusted
    values = p_values[valid]
    order = np.argsort(values)
    ranked = values[order]
    corrected = ranked * len(values) / np.arange(1, len(values) + 1)
    corrected = np.minimum.accumulate(corrected[::-1])[::-1]
    restored = np.empty(len(values))
    restored[order] = np.clip(corrected, 0, 1)
    adjusted[valid] = restored
    return adjusted


def read_mmse(path):
    table = pd.read_csv(path)
    table.columns = table.columns.str.strip()
    table = table.loc[:, ~table.columns.str.match(r"^Unnamed")].copy()
    required = ["subject_id", "MMSE_pre", "MMSE_post", "MMSE_3m"]
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise ValueError("Cognitive CSV is missing columns: " + ", ".join(missing))
    if table["subject_id"].isna().any() or not table["subject_id"].is_unique:
        raise ValueError("Cognitive subject_id must be non-missing and unique.")
    table["subject_id"] = table["subject_id"].astype(str).str.strip()
    wide = table[required].rename(columns={"MMSE_pre": "pre", "MMSE_post": "post", "MMSE_3m": "3m"})
    long = wide.melt(id_vars="subject_id", value_vars=TIMEPOINTS,
                     var_name="timepoint", value_name="MMSE")
    long["MMSE"] = pd.to_numeric(long["MMSE"], errors="coerce")
    return long


def read_csv(path, required):
    table = pd.read_csv(path)
    table.columns = table.columns.str.strip()
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
    return table


def read_family(long_path, omnibus_path, contrast_path):
    long = read_csv(long_path, ["subject_id", "Task", "ROI", "Measure", "timepoint", "value"])
    long = long[["subject_id", "Task", "ROI", "Measure", "timepoint", "value"]].copy()
    long["subject_id"] = long["subject_id"].astype(str).str.strip()
    for column in CELL_KEYS:
        long[column] = long[column].astype(str).str.strip()
    long["timepoint"] = long["timepoint"].astype(str).str.strip().str.lower()
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long[long["timepoint"].isin(TIMEPOINTS)]
    if long.duplicated(["subject_id", *CELL_KEYS, "timepoint"]).any():
        raise ValueError(f"{long_path} contains duplicate subject/cell/timepoint rows.")

    omnibus = read_csv(omnibus_path, [*CELL_KEYS, "p_adj_FDR"])
    contrast = read_csv(contrast_path, [*CELL_KEYS, "Contrast", "p_adj_FDR"])
    for table in (omnibus, contrast):
        for column in CELL_KEYS:
            table[column] = table[column].astype(str).str.strip()
        table["p_adj_FDR"] = pd.to_numeric(table["p_adj_FDR"], errors="coerce")
    if omnibus.duplicated(CELL_KEYS).any():
        raise ValueError(f"{omnibus_path} contains duplicate task/ROI/measure rows.")
    return long, omnibus[CELL_KEYS + ["p_adj_FDR"]], contrast[CELL_KEYS + ["Contrast", "p_adj_FDR"]]


def prepare_data(mmse, feature_long):
    data = mmse.merge(feature_long[["subject_id", "timepoint", "value"]],
                      on=["subject_id", "timepoint"], how="left", validate="one_to_one")
    data["timepoint"] = pd.Categorical(data["timepoint"], categories=TIMEPOINTS, ordered=True)
    person_mean = data.groupby("subject_id", observed=False)["value"].transform("mean")
    data["EEG_within"] = data["value"] - person_mean
    data["EEG_between"] = person_mean - data["value"].mean()
    return data


def fit_lmm(data):
    data = data.dropna(subset=["MMSE", "value", "EEG_within", "EEG_between"]).copy()
    counts = data.groupby("timepoint", observed=False)["MMSE"].count()
    if (counts < 2).any() or data["subject_id"].nunique() < 2:
        return None, "insufficient observations at one or more timepoints", data
    exog = pd.DataFrame({
        "Intercept": 1.0,
        "Month1": (data["timepoint"] == "post").astype(float).to_numpy(),
        "Month3": (data["timepoint"] == "3m").astype(float).to_numpy(),
        "EEG_within": data["EEG_within"].to_numpy(dtype=float),
        "EEG_between": data["EEG_between"].to_numpy(dtype=float),
    }, index=data.index)
    errors = []
    for method in ("lbfgs", "powell"):
        try:
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                fitted = sm.MixedLM(data["MMSE"], exog, groups=data["subject_id"]).fit(reml=True, method=method)
            if fitted.converged:
                return (fitted, " | ".join(str(item.message) for item in captured), data), None, data
            errors.append(f"{method}: did not converge")
        except Exception as error:
            errors.append(f"{method}: {type(error).__name__}: {error}")
    return None, "fit failed: " + " | ".join(errors), data


def model_rows(fitted, warnings_text, data, cell):
    terms = ["Intercept", "Month1", "Month3", "EEG_within", "EEG_between"]
    estimates = fitted.fe_params.loc[terms].to_numpy(dtype=float)
    covariance = fitted.cov_params().loc[terms, terms].to_numpy(dtype=float)
    standard_errors = np.sqrt(np.diag(covariance))
    z_values = estimates / standard_errors
    result = pd.DataFrame({
        **{column: cell[column] for column in CELL_KEYS},
        "EEG_omnibus_q_FDR": cell["EEG_omnibus_q_FDR"],
        "Term": terms,
        "Estimate": estimates,
        "SE": standard_errors,
        "CI_95_Low": estimates - 1.96 * standard_errors,
        "CI_95_High": estimates + 1.96 * standard_errors,
        "z_value": z_values,
        "p_value": 2 * stats.norm.sf(np.abs(z_values)),
        "N_subjects": data["subject_id"].nunique(),
        "N_observations": len(data),
        "N_0m": int((data["timepoint"] == "pre").sum()),
        "N_1m": int((data["timepoint"] == "post").sum()),
        "N_3m": int((data["timepoint"] == "3m").sum()),
        "Random_Intercept_Variance": float(fitted.cov_re.iloc[0, 0]),
        "Residual_SD": float(np.sqrt(fitted.scale)),
        "Fit_Warning": warnings_text,
    })
    return result


def spearman_summary(data, cell):
    pivot = data.pivot(index="subject_id", columns="timepoint", values=["MMSE", "value"])
    changes = pd.DataFrame({
        "subject_id": pivot.index.astype(str),
        **{column: cell[column] for column in CELL_KEYS},
        "EEG_3m_minus_0m_q_FDR": cell["EEG_3m_minus_0m_q_FDR"],
        "Delta_MMSE_3m_minus_0m": pivot[("MMSE", "3m")] - pivot[("MMSE", "pre")],
        "Delta_EEG_3m_minus_0m": pivot[("value", "3m")] - pivot[("value", "pre")],
    }).reset_index(drop=True)
    complete = changes.dropna(subset=["Delta_MMSE_3m_minus_0m", "Delta_EEG_3m_minus_0m"])
    if len(complete) < 3:
        rho, p_value = np.nan, np.nan
    else:
        rho, p_value = stats.spearmanr(complete["Delta_EEG_3m_minus_0m"], complete["Delta_MMSE_3m_minus_0m"])
    summary = {**{column: cell[column] for column in CELL_KEYS},
               "EEG_3m_minus_0m_q_FDR": cell["EEG_3m_minus_0m_q_FDR"],
               "N": len(complete), "Spearman_rho": rho, "p_value": p_value}
    return changes, summary


def rounded(table):
    table = table.copy()
    numeric = table.select_dtypes(include=np.number).columns
    table[numeric] = table[numeric].round(4)
    return table


def safe_filename(cell):
    text = "_".join(str(cell[column]) for column in CELL_KEYS)
    return "".join(character if character.isalnum() or character in "_-" else "-" for character in text)


def make_spearman_scatter(changes, result, out_path):
    complete = changes.dropna(subset=["Delta_MMSE_3m_minus_0m", "Delta_EEG_3m_minus_0m"])
    figure, axis = plt.subplots(figsize=(5.5, 4.5))
    x = complete["Delta_EEG_3m_minus_0m"]
    y = complete["Delta_MMSE_3m_minus_0m"]
    axis.scatter(x, y, color="#6A3D9A", edgecolor="white", linewidth=0.5, s=48)
    if len(complete) >= 2 and x.nunique() > 1:
        slope, intercept = np.polyfit(x, y, 1)
        x_values = np.linspace(x.min(), x.max(), 100)
        axis.plot(x_values, intercept + slope * x_values, color="black", linewidth=1.2, linestyle="--")
    axis.axhline(0, color="#999999", linewidth=0.8)
    axis.axvline(0, color="#999999", linewidth=0.8)
    axis.set_xlabel(f"Change in {result['Measure']} (3m − 0m)")
    axis.set_ylabel("Change in MMSE (3m − 0m)")
    axis.set_title(f"Task {result['Task']} | {result['ROI']}")
    axis.text(0.03, 0.97,
              f"Spearman ρ = {result['Spearman_rho']:.3f}\n"
              f"q-FDR = {result['q_FDR_spearman_across_selected_cells']:.3f}; n = {result['N']}",
              transform=axis.transAxes, ha="left", va="top", fontsize=9)
    figure.tight_layout()
    figure.savefig(out_path, dpi=300)
    plt.close(figure)


def make_lmm_within_scatter(data, result, out_path):
    complete = data.dropna(subset=["MMSE", "EEG_within"]).copy()
    covariates = np.column_stack((np.ones(len(complete)),
                                  (complete["timepoint"] == "post").astype(float),
                                  (complete["timepoint"] == "3m").astype(float)))
    x = complete["EEG_within"].to_numpy(dtype=float)
    y = complete["MMSE"].to_numpy(dtype=float)
    x_residual = x - covariates @ np.linalg.lstsq(covariates, x, rcond=None)[0]
    y_residual = y - covariates @ np.linalg.lstsq(covariates, y, rcond=None)[0]
    figure, axis = plt.subplots(figsize=(5.5, 4.5))
    axis.scatter(x_residual, y_residual, color="#2C7FB8", edgecolor="white", linewidth=0.5, s=48)
    if len(complete) >= 2 and np.unique(x_residual).size > 1:
        slope, intercept = np.polyfit(x_residual, y_residual, 1)
        x_values = np.linspace(x_residual.min(), x_residual.max(), 100)
        axis.plot(x_values, intercept + slope * x_values, color="black", linewidth=1.2, linestyle="--")
    axis.axhline(0, color="#999999", linewidth=0.8)
    axis.axvline(0, color="#999999", linewidth=0.8)
    axis.set_xlabel(f"Within-person {result['Measure']} (time-adjusted residual)")
    axis.set_ylabel("MMSE (time-adjusted residual)")
    axis.set_title(f"Task {result['Task']} | {result['ROI']}")
    axis.text(0.03, 0.97,
              f"LMM within-person β = {result['Estimate']:.3f}\n"
              f"q-FDR = {result['q_FDR_association_across_selected_cells']:.3f}; n = {result['N_observations']}",
              transform=axis.transAxes, ha="left", va="top", fontsize=9)
    figure.tight_layout()
    figure.savefig(out_path, dpi=300)
    plt.close(figure)


def main(cognitive_path, family_paths, output_dir):
    mmse = read_mmse(cognitive_path)
    all_long, all_omnibus, all_contrast = [], [], []
    for long_path, omnibus_path, contrast_path in family_paths:
        long, omnibus, contrast = read_family(long_path, omnibus_path, contrast_path)
        all_long.append(long)
        all_omnibus.append(omnibus)
        all_contrast.append(contrast)
    eeg_long = pd.concat(all_long, ignore_index=True)
    omnibus = pd.concat(all_omnibus, ignore_index=True).rename(columns={"p_adj_FDR": "EEG_omnibus_q_FDR"})
    contrast = pd.concat(all_contrast, ignore_index=True)
    if omnibus.duplicated(CELL_KEYS).any():
        raise ValueError("Duplicate task/ROI/measure cells across supplied family summaries.")

    output_dir.mkdir(parents=True, exist_ok=True)
    lmm_cells = omnibus[omnibus["EEG_omnibus_q_FDR"] < ALPHA].copy()
    lmm_cells.to_csv(output_dir / "selected_eeg_cells_omnibus_qFDR.csv", index=False)
    contrast_cells = contrast[(contrast["Contrast"] == "3m_minus_pre") & (contrast["p_adj_FDR"] < ALPHA)].copy()
    contrast_cells = contrast_cells.rename(columns={"p_adj_FDR": "EEG_3m_minus_0m_q_FDR"})
    contrast_cells = contrast_cells[CELL_KEYS + ["EEG_3m_minus_0m_q_FDR"]]
    contrast_cells.to_csv(output_dir / "selected_eeg_cells_3m_minus_0m_qFDR.csv", index=False)

    lmm_rows, lmm_failures = [], []
    for _, cell in lmm_cells.iterrows():
        mask = np.logical_and.reduce([eeg_long[column] == cell[column] for column in CELL_KEYS])
        data = prepare_data(mmse, eeg_long.loc[mask])
        fitted_result, status, model_data = fit_lmm(data)
        if fitted_result is None:
            lmm_failures.append({**{column: cell[column] for column in CELL_KEYS}, "Reason": status})
            continue
        fitted, warnings_text, model_data = fitted_result
        lmm_rows.append(model_rows(fitted, warnings_text, model_data, cell))
    lmm_results = pd.concat(lmm_rows, ignore_index=True) if lmm_rows else pd.DataFrame()
    if not lmm_results.empty:
        lmm_results["q_FDR_association_across_selected_cells"] = np.nan
        within = lmm_results["Term"] == "EEG_within"
        lmm_results.loc[within, "q_FDR_association_across_selected_cells"] = fdr_bh(
            lmm_results.loc[within, "p_value"].to_numpy()
        )
    rounded(lmm_results).to_csv(output_dir / "mmse_selected_eeg_association_lmm.csv", index=False)
    pd.DataFrame(lmm_failures, columns=[*CELL_KEYS, "Reason"]).to_csv(
        output_dir / "mmse_selected_eeg_association_lmm_failures.csv", index=False
    )
    lmm_figure_dir = output_dir / "figures_lmm_within_significant"
    lmm_significant = lmm_results.iloc[0:0].copy() if lmm_results.empty else lmm_results[
        (lmm_results["Term"] == "EEG_within")
        & (lmm_results["q_FDR_association_across_selected_cells"] < ALPHA)
    ]
    if not lmm_significant.empty:
        lmm_figure_dir.mkdir(exist_ok=True)
        for _, result in lmm_significant.iterrows():
            mask = np.logical_and.reduce([eeg_long[column] == result[column] for column in CELL_KEYS])
            data = prepare_data(mmse, eeg_long.loc[mask])
            make_lmm_within_scatter(data, result, lmm_figure_dir / f"{safe_filename(result)}_lmm_within.png")

    spearman_rows, change_rows, spearman_failures = [], [], []
    for _, cell in contrast_cells.iterrows():
        mask = np.logical_and.reduce([eeg_long[column] == cell[column] for column in CELL_KEYS])
        try:
            data = prepare_data(mmse, eeg_long.loc[mask])
            changes, summary = spearman_summary(data, cell)
            change_rows.append(changes)
            spearman_rows.append(summary)
        except Exception as error:
            spearman_failures.append({**{column: cell[column] for column in CELL_KEYS}, "Reason": str(error)})
    spearman_results = pd.DataFrame(spearman_rows)
    if not spearman_results.empty:
        spearman_results["q_FDR_spearman_across_selected_cells"] = fdr_bh(spearman_results["p_value"].to_numpy())
    rounded(spearman_results).to_csv(output_dir / "mmse_selected_eeg_spearman_0m_3m.csv", index=False)
    change_results = pd.concat(change_rows, ignore_index=True) if change_rows else pd.DataFrame()
    change_results.to_csv(
        output_dir / "mmse_selected_eeg_change_scores_0m_3m.csv", index=False
    )
    pd.DataFrame(spearman_failures, columns=[*CELL_KEYS, "Reason"]).to_csv(
        output_dir / "mmse_selected_eeg_spearman_failures.csv", index=False
    )
    spearman_figure_dir = output_dir / "figures_spearman_significant"
    spearman_significant = spearman_results.iloc[0:0].copy() if spearman_results.empty else spearman_results[
        spearman_results["q_FDR_spearman_across_selected_cells"] < ALPHA
    ]
    if not spearman_significant.empty:
        spearman_figure_dir.mkdir(exist_ok=True)
        for _, result in spearman_significant.iterrows():
            mask = np.logical_and.reduce([change_results[column] == result[column] for column in CELL_KEYS])
            make_spearman_scatter(change_results.loc[mask], result,
                                  spearman_figure_dir / f"{safe_filename(result)}_spearman_0m_3m.png")
    print(f"Selected {len(lmm_cells)} omnibus-significant EEG cells for LMM association.")
    print(f"Selected {len(contrast_cells)} 3m-minus-0m-significant EEG cells for Spearman analysis.")
    print(f"Saved {len(lmm_significant)} significant LMM association figure(s).")
    print(f"Saved {len(spearman_significant)} significant Spearman figure(s).")
    print(f"Saved results to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MMSE associations for selected longitudinal EEG cells.")
    parser.add_argument("cognitive_csv", help="Wide cognitive CSV containing MMSE_pre/post/3m")
    parser.add_argument("--eeg-family", action="append", nargs=3, required=True,
                        metavar=("LONG_CSV", "OMNIBUS_CSV", "CONTRAST_CSV"),
                        help="EEG family long, omnibus summary, and contrast summary CSV paths; repeat per family")
    parser.add_argument("--output-dir", default="results_mmse_selected_eeg_associations",
                        help="Output directory (default: results_mmse_selected_eeg_associations)")
    arguments = parser.parse_args()
    try:
        main(arguments.cognitive_csv, arguments.eeg_family, Path(arguments.output_dir))
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
