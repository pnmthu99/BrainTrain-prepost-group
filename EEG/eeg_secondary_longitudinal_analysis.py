# -*- coding: utf-8 -*-
"""Secondary longitudinal EEG analyses using ROI and global composites.

Inputs are the *_long.csv files exported by eeg_feature_family_lmm.py. The
script retains each measure separately and creates:

* task composites: mean across the five ROIs;
* global composites: mean across all tasks and ROIs;
* global and task-composite time-effect LMMs;
* one time-by-task interaction LMM per measure;
* individual trajectory plots for global composites with q-FDR < .05;
* availability and baseline-completer sensitivity tables.
* optionally, exploratory baseline EEG predictors of 0m-to-3m MMSE change.

The original task x ROI analyses remain the cell-level supplementary results.

Example
-------
python3 eeg_secondary_longitudinal_analysis.py \
  --eeg-long results_entropy_lmm/entropy_long.csv \
  --eeg-long results_relative_power_lmm/relative_power_long.csv \
  --eeg-long results_theta_alpha_ratio_lmm/theta_alpha_ratio_long.csv \
  --eeg-long results_theta_beta_ratio_lmm/theta_beta_ratio_long.csv \
  --cognitive-csv ../prepost-comparison/cogtest_013m.csv \
  --output-dir results_eeg_secondary_longitudinal
"""

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

try:
    import statsmodels.api as sm
except ImportError as error:
    raise SystemExit("This script requires statsmodels: pip install statsmodels") from error


ALPHA = 0.05
TIMEPOINTS = ["pre", "post", "3m"]
TIME_LABELS = {"pre": "0m", "post": "1m", "3m": "3m"}
CELL_KEYS = ["Task", "ROI", "Family", "Measure"]


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
    restored = np.empty(len(values))
    restored[order] = np.minimum.accumulate(corrected[::-1])[::-1].clip(0, 1)
    adjusted[valid] = restored
    return adjusted


def read_eeg_long(paths):
    frames = []
    required = ["subject_id", "Task", "ROI", "Family", "Measure", "timepoint", "value"]
    for path in paths:
        table = pd.read_csv(path)
        table.columns = table.columns.str.strip()
        missing = [column for column in required if column not in table.columns]
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
        table = table[required].copy()
        for column in ["subject_id", "Task", "ROI", "Family", "Measure"]:
            table[column] = table[column].astype(str).str.strip()
        table["timepoint"] = table["timepoint"].astype(str).str.strip().str.lower()
        table["value"] = pd.to_numeric(table["value"], errors="coerce")
        table = table[table["timepoint"].isin(TIMEPOINTS)]
        if table.duplicated(["subject_id", *CELL_KEYS, "timepoint"]).any():
            raise ValueError(f"{path} contains duplicate subject/cell/timepoint rows.")
        frames.append(table)
    long = pd.concat(frames, ignore_index=True)
    if long.duplicated(["subject_id", *CELL_KEYS, "timepoint"]).any():
        raise ValueError("The supplied long files overlap on subject/cell/timepoint rows.")
    return long


def mixed_fit(data, exog):
    errors = []
    for method in ("lbfgs", "powell"):
        try:
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                result = sm.MixedLM(data["value"], exog, groups=data["subject_id"]).fit(
                    reml=False, method=method
                )
            if result.converged:
                warning_text = " | ".join(str(item.message) for item in captured)
                return result, warning_text, None
            errors.append(f"{method}: did not converge")
        except Exception as error:
            errors.append(f"{method}: {type(error).__name__}: {error}")
    return None, "", "fit failed: " + " | ".join(errors)


def time_design(data, include_time=True):
    design = {"Intercept": np.ones(len(data))}
    if include_time:
        design["Month1"] = (data["timepoint"] == "post").astype(float).to_numpy()
        design["Month3"] = (data["timepoint"] == "3m").astype(float).to_numpy()
    return pd.DataFrame(design, index=data.index)


def lrt_time_effect(data):
    data = data.dropna(subset=["value"]).copy()
    counts = data.groupby("timepoint", observed=False)["value"].count()
    if (counts.reindex(TIMEPOINTS, fill_value=0) < 2).any() or data["subject_id"].nunique() < 2:
        return None, "insufficient observations at one or more timepoints"
    reduced, _, error = mixed_fit(data, time_design(data, include_time=False))
    if error:
        return None, "reduced model " + error
    full, warning_text, error = mixed_fit(data, time_design(data, include_time=True))
    if error:
        return None, "full model " + error
    statistic = max(0.0, 2 * (full.llf - reduced.llf))
    p_value = stats.chi2.sf(statistic, 2)
    terms = full.fe_params
    return {
        "LRT_chi2": statistic,
        "df": 2,
        "p_value": p_value,
        "Estimate_1m_minus_0m": terms["Month1"],
        "Estimate_3m_minus_0m": terms["Month3"],
        "N_subjects": data["subject_id"].nunique(),
        "N_observations": len(data),
        "N_0m": int((data["timepoint"] == "pre").sum()),
        "N_1m": int((data["timepoint"] == "post").sum()),
        "N_3m": int((data["timepoint"] == "3m").sum()),
        "Random_Intercept_Variance": float(full.cov_re.iloc[0, 0]),
        "Residual_SD": float(np.sqrt(full.scale)),
        "Fit_Warning": warning_text,
    }, None


def interaction_design(data, include_interaction=True):
    tasks = sorted(data["Task"].unique())
    reference = tasks[0]
    month1 = (data["timepoint"] == "post").astype(float).to_numpy()
    month3 = (data["timepoint"] == "3m").astype(float).to_numpy()
    design = {"Intercept": np.ones(len(data)), "Month1": month1, "Month3": month3}
    for task in tasks[1:]:
        task_indicator = (data["Task"] == task).astype(float).to_numpy()
        design[f"Task_{task}"] = task_indicator
        if include_interaction:
            design[f"Month1_x_Task_{task}"] = month1 * task_indicator
            design[f"Month3_x_Task_{task}"] = month3 * task_indicator
    return pd.DataFrame(design, index=data.index), tasks, reference


def lrt_time_by_task(data):
    data = data.dropna(subset=["value"]).copy()
    if data["Task"].nunique() < 2 or data["subject_id"].nunique() < 2:
        return None, "insufficient task or subject variation"
    reduced_design, tasks, reference = interaction_design(data, include_interaction=False)
    full_design, _, _ = interaction_design(data, include_interaction=True)
    reduced, _, error = mixed_fit(data, reduced_design)
    if error:
        return None, "reduced model " + error
    full, warning_text, error = mixed_fit(data, full_design)
    if error:
        return None, "full model " + error
    degrees = 2 * (len(tasks) - 1)
    statistic = max(0.0, 2 * (full.llf - reduced.llf))
    return {
        "Reference_Task": reference,
        "LRT_chi2": statistic,
        "df": degrees,
        "p_value": stats.chi2.sf(statistic, degrees),
        "N_subjects": data["subject_id"].nunique(),
        "N_observations": len(data),
        "Fit_Warning": warning_text,
    }, None


def make_composites(long):
    task = long.groupby(["subject_id", "Task", "Family", "Measure", "timepoint"], as_index=False,
                        observed=False)["value"].mean()
    task["Composite"] = "ROI_mean"
    global_ = long.groupby(["subject_id", "Family", "Measure", "timepoint"], as_index=False,
                           observed=False)["value"].mean()
    global_["Composite"] = "Task_ROI_mean"
    return task, global_


def fit_composite_time(composite, grouping_columns):
    rows, failures = [], []
    for keys, data in composite.groupby(grouping_columns, observed=False):
        keys = (keys,) if not isinstance(keys, tuple) else keys
        metadata = dict(zip(grouping_columns, keys))
        result, error = lrt_time_effect(data)
        if error:
            failures.append({**metadata, "Reason": error})
        else:
            rows.append({**metadata, **result})
    results = pd.DataFrame(rows)
    if not results.empty:
        results["q_FDR_time_effect"] = fdr_bh(results["p_value"])
        results["Significant_after_FDR"] = results["q_FDR_time_effect"] < ALPHA
    return results, pd.DataFrame(failures)


def fit_task_interactions(task_composite):
    rows, failures = [], []
    for keys, data in task_composite.groupby(["Family", "Measure"], observed=False):
        family, measure = keys
        result, error = lrt_time_by_task(data)
        if error:
            failures.append({"Family": family, "Measure": measure, "Reason": error})
        else:
            rows.append({"Family": family, "Measure": measure, **result})
    results = pd.DataFrame(rows)
    if not results.empty:
        results["q_FDR_time_by_task"] = fdr_bh(results["p_value"])
        results["Significant_after_FDR"] = results["q_FDR_time_by_task"] < ALPHA
    return results, pd.DataFrame(failures)


def availability(long):
    all_subjects = long["subject_id"].nunique()
    result = long.groupby(["Family", "Measure", "Task", "ROI", "timepoint"], observed=False)["value"].agg(
        N_available="count", Mean="mean", SD="std"
    ).reset_index()
    result["N_total_subjects"] = all_subjects
    result["N_missing"] = all_subjects - result["N_available"]
    return result


def baseline_completer_sensitivity(global_composite):
    rows = []
    for (family, measure), data in global_composite.groupby(["Family", "Measure"], observed=False):
        pivot = data.pivot(index="subject_id", columns="timepoint", values="value").reindex(columns=TIMEPOINTS)
        baseline = pivot["pre"]
        complete_3m = baseline.notna() & pivot["3m"].notna()
        completers = baseline[complete_3m].dropna()
        non_completers = baseline[(~complete_3m) & baseline.notna()]
        if len(completers) >= 2 and len(non_completers) >= 2:
            statistic, p_value = stats.ttest_ind(completers, non_completers, equal_var=False)
        else:
            statistic, p_value = np.nan, np.nan
        rows.append({
            "Family": family, "Measure": measure,
            "N_3m_completers": int(complete_3m.sum()),
            "N_not_3m_complete": int((~complete_3m).sum()),
            "Baseline_Mean_3m_completers": completers.mean(),
            "Baseline_Mean_not_3m_complete": non_completers.mean(),
            "Welch_t": statistic, "p_value": p_value,
        })
    result = pd.DataFrame(rows)
    if not result.empty:
        result["q_FDR_baseline_completer_comparison"] = fdr_bh(result["p_value"])
    return result


def read_mmse_change(path):
    table = pd.read_csv(path)
    table.columns = table.columns.str.strip()
    required = ["subject_id", "MMSE_pre", "MMSE_3m"]
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
    data = table[required].copy()
    if data["subject_id"].isna().any() or data["subject_id"].duplicated().any():
        raise ValueError("Cognitive subject_id must be non-missing and unique.")
    data["subject_id"] = data["subject_id"].astype(str).str.strip()
    data["MMSE_pre"] = pd.to_numeric(data["MMSE_pre"], errors="coerce")
    data["MMSE_3m"] = pd.to_numeric(data["MMSE_3m"], errors="coerce")
    data["Delta_MMSE_3m_minus_0m"] = data["MMSE_3m"] - data["MMSE_pre"]
    return data[["subject_id", "MMSE_pre", "Delta_MMSE_3m_minus_0m"]]


def baseline_eeg_mmse_predictors(global_composite, mmse):
    rows = []
    baseline = global_composite[global_composite["timepoint"] == "pre"]
    for (family, measure), data in baseline.groupby(["Family", "Measure"], observed=False):
        model_data = data[["subject_id", "value"]].merge(mmse, on="subject_id", how="inner").dropna()
        if len(model_data) < 6 or model_data["value"].nunique() < 2:
            rows.append({"Family": family, "Measure": measure, "N": len(model_data),
                         "Reason": "insufficient complete or variable data"})
            continue
        design = sm.add_constant(model_data[["MMSE_pre", "value"]], has_constant="add")
        fitted = sm.OLS(model_data["Delta_MMSE_3m_minus_0m"], design).fit()
        rows.append({
            "Family": family, "Measure": measure, "N": len(model_data),
            "EEG_0m_Estimate": fitted.params["value"],
            "EEG_0m_SE": fitted.bse["value"],
            "EEG_0m_t": fitted.tvalues["value"],
            "p_value": fitted.pvalues["value"],
            "Baseline_MMSE_Estimate": fitted.params["MMSE_pre"],
            "Model_R_squared": fitted.rsquared,
            "Reason": "",
        })
    results = pd.DataFrame(rows)
    if "p_value" in results:
        results["q_FDR_across_global_measures"] = fdr_bh(results["p_value"])
    return results


def safe_filename(family, measure):
    text = f"{family}_{measure}"
    return "".join(character if character.isalnum() or character in "_-" else "-" for character in text)


def plot_global_trajectory(data, result, out_path):
    figure, axis = plt.subplots(figsize=(6.2, 4.6))
    ordered = data.copy()
    ordered["timepoint"] = pd.Categorical(ordered["timepoint"], categories=TIMEPOINTS, ordered=True)
    for _, subject_data in ordered.groupby("subject_id", observed=False):
        subject_data = subject_data.sort_values("timepoint")
        axis.plot(range(3), subject_data.set_index("timepoint").reindex(TIMEPOINTS)["value"],
                  color="#BFBFBF", linewidth=0.8, alpha=0.7, zorder=1)
    summary = ordered.groupby("timepoint", observed=False)["value"].agg(["mean", "count", "std"]).reindex(TIMEPOINTS)
    summary["sem"] = summary["std"] / np.sqrt(summary["count"])
    axis.errorbar(range(3), summary["mean"], yerr=1.96 * summary["sem"], color="black", marker="o",
                  markersize=5, linewidth=2, capsize=3, label="Mean ± 95% CI", zorder=3)
    axis.set_xticks(range(3), [TIME_LABELS[item] for item in TIMEPOINTS])
    axis.set_xlabel("Timepoint")
    axis.set_ylabel(result["Measure"])
    axis.set_title(f"Global {result['Measure']}")
    axis.text(0.03, 0.97, f"Time effect: q-FDR = {result['q_FDR_time_effect']:.3f}",
              transform=axis.transAxes, ha="left", va="top", fontsize=9)
    axis.legend(frameon=False, fontsize=8, loc="best")
    figure.tight_layout()
    figure.savefig(out_path, dpi=300)
    plt.close(figure)


def rounded(table):
    table = table.copy()
    numeric = table.select_dtypes(include=np.number).columns
    table[numeric] = table[numeric].round(4)
    return table


def main(long_paths, output_dir, cognitive_path=None):
    long = read_eeg_long(long_paths)
    output_dir.mkdir(parents=True, exist_ok=True)
    task_composite, global_composite = make_composites(long)
    global_results, global_failures = fit_composite_time(global_composite, ["Family", "Measure"])
    task_results, task_failures = fit_composite_time(task_composite, ["Family", "Measure", "Task"])
    interactions, interaction_failures = fit_task_interactions(task_composite)

    long.to_csv(output_dir / "eeg_cell_level_long_combined.csv", index=False)
    task_composite.to_csv(output_dir / "eeg_task_roi_mean_long.csv", index=False)
    global_composite.to_csv(output_dir / "eeg_global_task_roi_mean_long.csv", index=False)
    rounded(global_results).to_csv(output_dir / "eeg_global_composite_time_lmm.csv", index=False)
    rounded(task_results).to_csv(output_dir / "eeg_task_composite_time_lmm.csv", index=False)
    rounded(interactions).to_csv(output_dir / "eeg_time_by_task_interaction_lmm.csv", index=False)
    rounded(availability(long)).to_csv(output_dir / "eeg_data_availability.csv", index=False)
    rounded(baseline_completer_sensitivity(global_composite)).to_csv(
        output_dir / "eeg_baseline_3m_completer_sensitivity.csv", index=False
    )
    if cognitive_path:
        mmse = read_mmse_change(cognitive_path)
        rounded(baseline_eeg_mmse_predictors(global_composite, mmse)).to_csv(
            output_dir / "exploratory_eeg_0m_predictors_of_mmse_change.csv", index=False
        )
    pd.concat([global_failures.assign(Analysis="global composite time"),
               task_failures.assign(Analysis="task composite time"),
               interaction_failures.assign(Analysis="time by task interaction")],
              ignore_index=True, sort=False).to_csv(output_dir / "eeg_secondary_lmm_failures.csv", index=False)

    figure_dir = output_dir / "figures_global_composite_significant"
    significant = global_results[global_results["q_FDR_time_effect"] < ALPHA] if not global_results.empty else pd.DataFrame()
    if not significant.empty:
        figure_dir.mkdir(exist_ok=True)
        for _, result in significant.iterrows():
            data = global_composite[(global_composite["Family"] == result["Family"])
                                    & (global_composite["Measure"] == result["Measure"])]
            plot_global_trajectory(data, result, figure_dir / f"{safe_filename(result['Family'], result['Measure'])}.png")
    print(f"Loaded {long['subject_id'].nunique()} subjects and {len(long)} EEG cell-level rows.")
    print(f"Saved {len(significant)} significant global-composite trajectory figure(s).")
    print(f"Saved results to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Secondary longitudinal EEG composite analyses.")
    parser.add_argument("--eeg-long", action="append", required=True,
                        help="A *_long.csv file exported by eeg_feature_family_lmm.py; repeat per family.")
    parser.add_argument("--output-dir", default="results_eeg_secondary_longitudinal",
                        help="Output directory (default: results_eeg_secondary_longitudinal)")
    parser.add_argument("--cognitive-csv",
                        help="Optional cogtest_013m.csv for exploratory EEG-0m predictors of MMSE change")
    arguments = parser.parse_args()
    try:
        main(arguments.eeg_long, Path(arguments.output_dir), arguments.cognitive_csv)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
