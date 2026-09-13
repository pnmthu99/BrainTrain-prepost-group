# -*- coding: utf-8 -*-
"""Planned MMSE associations across all longitudinal EEG feature measures.

For each EEG measure, the script creates:
  1. a global task-state composite (mean across all available task x ROI cells);
  2. four task-specific composites (mean across the five ROIs per task).

Each composite is analysed with:
    MMSE ~ timepoint + EEG_within + EEG_between + (1 | subject)

The within-person EEG coefficient is the principal result.  All seven global
within-person coefficients are FDR-adjusted together.  The 28 task-specific
within-person coefficients are FDR-adjusted together.  Spearman correlations
of 0m-to-3m change scores are exported as secondary analyses with matching
FDR families.

Usage
-----
    python3 mmse_eeg_feature_associations.py cogtest_013m.csv \
        results_entropy_lmm/entropy_long.csv \
        results_relative_power_lmm/relative_power_long.csv \
        results_theta_alpha_ratio_lmm/theta_alpha_ratio_long.csv \
        results_theta_beta_ratio_lmm/theta_beta_ratio_long.csv
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

try:
    import statsmodels.api as sm
except ImportError as error:
    raise SystemExit("This script requires statsmodels: pip install statsmodels") from error


TIMEPOINTS = ["pre", "post", "3m"]


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


def read_eeg_longs(paths):
    frames = []
    required = ["subject_id", "Task", "ROI", "Measure", "timepoint", "value"]
    for path in paths:
        table = pd.read_csv(path)
        table.columns = table.columns.str.strip()
        missing = [column for column in required if column not in table.columns]
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
        table = table[required].copy()
        table["subject_id"] = table["subject_id"].astype(str).str.strip()
        table["Task"] = table["Task"].astype(str).str.strip()
        table["ROI"] = table["ROI"].astype(str).str.strip()
        table["Measure"] = table["Measure"].astype(str).str.strip()
        table["timepoint"] = table["timepoint"].astype(str).str.strip().str.lower()
        table["value"] = pd.to_numeric(table["value"], errors="coerce")
        frames.append(table[table["timepoint"].isin(TIMEPOINTS)])
    eeg = pd.concat(frames, ignore_index=True)
    if eeg.empty:
        raise ValueError("No EEG rows at pre, post or 3m were found.")
    keys = ["subject_id", "Task", "ROI", "Measure", "timepoint"]
    if eeg.duplicated(keys).any():
        raise ValueError("Duplicate subject/task/ROI/measure/timepoint rows across EEG long CSV files.")
    return eeg


def make_composites(eeg):
    global_composite = eeg.groupby(["subject_id", "timepoint", "Measure"], as_index=False)["value"].agg(
        EEG_value="mean", N_source_features="count"
    )
    task_composite = eeg.groupby(["subject_id", "timepoint", "Task", "Measure"], as_index=False)["value"].agg(
        EEG_value="mean", N_source_features="count"
    )
    return global_composite, task_composite


def prepare_data(mmse, composite):
    data = mmse.merge(composite, on=["subject_id", "timepoint"], how="left", validate="one_to_one")
    data["timepoint"] = pd.Categorical(data["timepoint"], categories=TIMEPOINTS, ordered=True)
    person_mean = data.groupby("subject_id", observed=False)["EEG_value"].transform("mean")
    data["EEG_within"] = data["EEG_value"] - person_mean
    data["EEG_between"] = person_mean - data["EEG_value"].mean()
    return data


def fit_association(data):
    data = data.dropna(subset=["MMSE", "EEG_value", "EEG_within", "EEG_between"]).copy()
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


def model_terms(fitted, warnings_text, data, measure, task, analysis_level):
    terms = ["Intercept", "Month1", "Month3", "EEG_within", "EEG_between"]
    estimates = fitted.fe_params.loc[terms].to_numpy(dtype=float)
    covariance = fitted.cov_params().loc[terms, terms].to_numpy(dtype=float)
    standard_errors = np.sqrt(np.diag(covariance))
    z_values = estimates / standard_errors
    table = pd.DataFrame({
        "Measure": measure,
        "Task": task,
        "Analysis_level": analysis_level,
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
    return table


def spearman_changes(data, measure, task, analysis_level):
    pivot = data.pivot(index="subject_id", columns="timepoint", values=["MMSE", "EEG_value"])
    changes = pd.DataFrame({
        "subject_id": pivot.index.astype(str),
        "Measure": measure,
        "Task": task,
        "Analysis_level": analysis_level,
        "Delta_MMSE_3m_minus_0m": pivot[("MMSE", "3m")] - pivot[("MMSE", "pre")],
        "Delta_EEG_3m_minus_0m": pivot[("EEG_value", "3m")] - pivot[("EEG_value", "pre")],
    }).reset_index(drop=True)
    complete = changes.dropna(subset=["Delta_MMSE_3m_minus_0m", "Delta_EEG_3m_minus_0m"])
    if len(complete) < 3:
        rho, p_value = np.nan, np.nan
    else:
        rho, p_value = stats.spearmanr(complete["Delta_EEG_3m_minus_0m"], complete["Delta_MMSE_3m_minus_0m"])
    summary = {"Measure": measure, "Task": task, "Analysis_level": analysis_level,
               "N": len(complete), "Spearman_rho": rho, "p_value": p_value}
    return changes, summary


def apply_within_fdr(table):
    if table.empty:
        return table
    table = table.copy()
    table["q_FDR_within_across_family"] = np.nan
    mask = table["Term"] == "EEG_within"
    table.loc[mask, "q_FDR_within_across_family"] = fdr_bh(table.loc[mask, "p_value"].to_numpy())
    return table


def analyse_composites(mmse, composites, level):
    model_rows, spearman_rows, change_rows, failures = [], [], [], []
    group_columns = ["Measure"] if level == "global" else ["Measure", "Task"]
    for keys, composite in composites.groupby(group_columns, observed=False):
        if level == "global":
            # pandas versions differ: a one-column group key may be returned
            # as either a scalar or a one-item tuple.
            measure = keys[0] if isinstance(keys, tuple) else keys
            task = "All tasks"
        else:
            measure, task = keys
        data = prepare_data(mmse, composite.drop(columns=["Measure", "Task"], errors="ignore"))
        fitted_result, status, model_data = fit_association(data)
        if fitted_result is None:
            failures.append({"Measure": measure, "Task": task, "Analysis_level": level, "Reason": status})
        else:
            fitted, warnings_text, model_data = fitted_result
            model_rows.append(model_terms(fitted, warnings_text, model_data, measure, task, level))
        changes, spearman = spearman_changes(data, measure, task, level)
        change_rows.append(changes)
        spearman_rows.append(spearman)
    models = apply_within_fdr(pd.concat(model_rows, ignore_index=True) if model_rows else pd.DataFrame())
    spearman = pd.DataFrame(spearman_rows)
    if not spearman.empty:
        spearman["q_FDR_across_family"] = fdr_bh(spearman["p_value"].to_numpy())
    changes = pd.concat(change_rows, ignore_index=True) if change_rows else pd.DataFrame()
    return models, spearman, changes, pd.DataFrame(failures)


def rounded(table):
    table = table.copy()
    numeric = table.select_dtypes(include=np.number).columns
    table[numeric] = table[numeric].round(4)
    return table


def main(cognitive_path, eeg_paths, output_dir):
    mmse = read_mmse(cognitive_path)
    eeg = read_eeg_longs(eeg_paths)
    global_composite, task_composite = make_composites(eeg)
    output_dir.mkdir(parents=True, exist_ok=True)
    global_composite.to_csv(output_dir / "mmse_eeg_global_composite_long.csv", index=False)
    task_composite.to_csv(output_dir / "mmse_eeg_task_composite_long.csv", index=False)

    global_results = analyse_composites(mmse, global_composite, "global")
    task_results = analyse_composites(mmse, task_composite, "task_specific")
    for prefix, results in (("mmse_eeg_global", global_results), ("mmse_eeg_task", task_results)):
        models, spearman, changes, failures = results
        rounded(models).to_csv(output_dir / f"{prefix}_association_lmm.csv", index=False)
        rounded(spearman).to_csv(output_dir / f"{prefix}_spearman_0m_3m.csv", index=False)
        changes.to_csv(output_dir / f"{prefix}_change_scores_0m_3m.csv", index=False)
        failures.to_csv(output_dir / f"{prefix}_association_failures.csv", index=False)

    print(f"Loaded {len(eeg)} EEG rows across {eeg['Measure'].nunique()} measures.")
    print(f"Saved results to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MMSE associations across planned EEG feature measures.")
    parser.add_argument("cognitive_csv", help="Wide cognitive CSV containing MMSE_pre/post/3m")
    parser.add_argument("eeg_long_csv", nargs="+", help="One or more EEG family *_long.csv files")
    parser.add_argument("--output-dir", default="results_mmse_eeg_associations",
                        help="Output directory (default: results_mmse_eeg_associations)")
    arguments = parser.parse_args()
    try:
        main(arguments.cognitive_csv, arguments.eeg_long_csv, Path(arguments.output_dir))
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
