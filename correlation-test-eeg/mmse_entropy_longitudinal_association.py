# -*- coding: utf-8 -*-
"""Longitudinal association between MMSE and global task-state EEG entropy.

This is the pre-specified brain-behaviour follow-up after the separate MMSE
and entropy time-effect analyses.  It reads the cognitive wide CSV and the
entropy_long.csv exported by eeg_feature_family_lmm.py.  At each subject and
timepoint, global entropy is the mean of all available task x ROI entropy
values.  The output records the number of values contributing to every mean.

Primary model
-------------
    MMSE ~ timepoint + entropy_within + entropy_between + (1 | subject)

entropy_within is the session entropy minus that participant's mean entropy.
Its coefficient tests whether longitudinal entropy deviations within a person
are associated with concurrent MMSE deviations after accounting for timepoint.
entropy_between is the participant mean entropy minus the cohort mean and is
included only to separate stable between-person differences from the within-
person association.

Secondary visualization
-----------------------
Spearman correlation of 0m-to-3m changes in MMSE and global entropy.  This is
an intuitive sensitivity analysis, not the primary longitudinal inference.

Secondary task-specific analysis
--------------------------------
For each task, entropy is averaged across the five ROIs and the same
within/between-person LMM and 0m-to-3m Spearman analysis are run.  The four
within-person LMM p-values and the four Spearman p-values are each
Benjamini-Hochberg FDR-adjusted across tasks.

Usage
-----
    python3 mmse_entropy_longitudinal_association.py \
        cogtest_013m.csv results_entropy_lmm/entropy_long.csv
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
    raise SystemExit(
        "This script requires matplotlib. Install it in the Python environment "
        "used to run the script: pip install matplotlib"
    ) from error

try:
    import statsmodels.api as sm
except ImportError as error:
    raise SystemExit(
        "This script requires statsmodels. Install it in the Python environment "
        "used to run the script: pip install statsmodels"
    ) from error


TIMEPOINTS = ["pre", "post", "3m"]
TIME_LABELS = {"pre": "0m", "post": "1m", "3m": "3m"}


def read_cognitive(path):
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
    wide = table[["subject_id", "MMSE_pre", "MMSE_post", "MMSE_3m"]].copy()
    wide = wide.rename(columns={"MMSE_pre": "pre", "MMSE_post": "post", "MMSE_3m": "3m"})
    long = wide.melt(id_vars="subject_id", value_vars=TIMEPOINTS,
                     var_name="timepoint", value_name="MMSE")
    long["MMSE"] = pd.to_numeric(long["MMSE"], errors="coerce")
    return long


def read_global_entropy(path):
    table = pd.read_csv(path)
    table.columns = table.columns.str.strip()
    required = ["subject_id", "Task", "ROI", "Measure", "timepoint", "value"]
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise ValueError("Entropy long CSV is missing columns: " + ", ".join(missing))
    table["subject_id"] = table["subject_id"].astype(str).str.strip()
    table["timepoint"] = table["timepoint"].astype(str).str.strip().str.lower()
    entropy = table[table["Measure"].astype(str).str.strip() == "entropy"].copy()
    entropy = entropy[entropy["timepoint"].isin(TIMEPOINTS)]
    entropy["value"] = pd.to_numeric(entropy["value"], errors="coerce")
    if entropy.empty:
        raise ValueError("No entropy rows at pre, post or 3m were found in entropy_long.csv.")
    duplicate_keys = entropy.duplicated(["subject_id", "Task", "ROI", "timepoint"])
    if duplicate_keys.any():
        raise ValueError("Entropy long CSV contains duplicate subject/task/ROI/timepoint rows.")
    composite = entropy.groupby(["subject_id", "timepoint"], as_index=False)["value"].agg(
        Global_Entropy="mean", N_entropy_features="count"
    )
    return composite


def read_task_entropy(path):
    table = pd.read_csv(path)
    table.columns = table.columns.str.strip()
    required = ["subject_id", "Task", "ROI", "Measure", "timepoint", "value"]
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise ValueError("Entropy long CSV is missing columns: " + ", ".join(missing))
    table["subject_id"] = table["subject_id"].astype(str).str.strip()
    table["timepoint"] = table["timepoint"].astype(str).str.strip().str.lower()
    entropy = table[table["Measure"].astype(str).str.strip() == "entropy"].copy()
    entropy = entropy[entropy["timepoint"].isin(TIMEPOINTS)]
    entropy["value"] = pd.to_numeric(entropy["value"], errors="coerce")
    if entropy.empty:
        raise ValueError("No entropy rows at pre, post or 3m were found in entropy_long.csv.")
    duplicate_keys = entropy.duplicated(["subject_id", "Task", "ROI", "timepoint"])
    if duplicate_keys.any():
        raise ValueError("Entropy long CSV contains duplicate subject/task/ROI/timepoint rows.")
    return entropy.groupby(["subject_id", "timepoint", "Task"], as_index=False)["value"].agg(
        Task_Entropy="mean", N_entropy_features="count"
    )


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


def prepare_model_data(cognitive, entropy):
    data = cognitive.merge(entropy, on=["subject_id", "timepoint"], how="left", validate="one_to_one")
    data["timepoint"] = pd.Categorical(data["timepoint"], categories=TIMEPOINTS, ordered=True)
    data["Entropy_person_mean"] = data.groupby("subject_id", observed=False)["Global_Entropy"].transform("mean")
    cohort_mean = data["Global_Entropy"].mean()
    data["Entropy_within"] = data["Global_Entropy"] - data["Entropy_person_mean"]
    data["Entropy_between"] = data["Entropy_person_mean"] - cohort_mean
    return data


def design_matrix(data):
    return pd.DataFrame({
        "Intercept": 1.0,
        "Month1": (data["timepoint"] == "post").astype(float).to_numpy(),
        "Month3": (data["timepoint"] == "3m").astype(float).to_numpy(),
        "Entropy_within": data["Entropy_within"].to_numpy(dtype=float),
        "Entropy_between": data["Entropy_between"].to_numpy(dtype=float),
    }, index=data.index)


def fit_lmm(data):
    model_data = data.dropna(subset=["MMSE", "Global_Entropy", "Entropy_within", "Entropy_between"]).copy()
    counts = model_data.groupby("timepoint", observed=False)["MMSE"].count()
    if (counts < 2).any() or model_data["subject_id"].nunique() < 2:
        return None, "insufficient observations at one or more timepoints", model_data
    exog = design_matrix(model_data)
    errors = []
    for method in ("lbfgs", "powell"):
        try:
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                fitted = sm.MixedLM(model_data["MMSE"], exog, groups=model_data["subject_id"]).fit(
                    reml=True, method=method
                )
            if fitted.converged:
                warnings_text = " | ".join(str(item.message) for item in captured)
                return (fitted, warnings_text, model_data), None, model_data
            errors.append(f"{method}: did not converge")
        except Exception as error:
            errors.append(f"{method}: {type(error).__name__}: {error}")
    return None, "fit failed: " + " | ".join(errors), model_data


def model_table(fitted, fit_warnings, model_data):
    terms = ["Intercept", "Month1", "Month3", "Entropy_within", "Entropy_between"]
    fixed = fitted.fe_params.loc[terms]
    covariance = fitted.cov_params().loc[terms, terms]
    standard_errors = np.sqrt(np.diag(covariance))
    z_values = fixed.to_numpy() / standard_errors
    p_values = 2 * stats.norm.sf(np.abs(z_values))
    result = pd.DataFrame({
        "Term": terms,
        "Estimate": fixed.to_numpy(),
        "SE": standard_errors,
        "CI_95_Low": fixed.to_numpy() - 1.96 * standard_errors,
        "CI_95_High": fixed.to_numpy() + 1.96 * standard_errors,
        "z_value": z_values,
        "p_value": p_values,
    })
    result["N_subjects"] = model_data["subject_id"].nunique()
    result["N_observations"] = len(model_data)
    result["N_0m"] = int((model_data["timepoint"] == "pre").sum())
    result["N_1m"] = int((model_data["timepoint"] == "post").sum())
    result["N_3m"] = int((model_data["timepoint"] == "3m").sum())
    result["Random_Intercept_Variance"] = float(fitted.cov_re.iloc[0, 0])
    result["Residual_SD"] = float(np.sqrt(fitted.scale))
    result["Fit_Warning"] = fit_warnings
    return result


def spearman_change(data):
    pivot = data.pivot(index="subject_id", columns="timepoint", values=["MMSE", "Global_Entropy"])
    changes = pd.DataFrame({
        "subject_id": pivot.index.astype(str),
        "Delta_MMSE_3m_minus_0m": pivot[("MMSE", "3m")] - pivot[("MMSE", "pre")],
        "Delta_Global_Entropy_3m_minus_0m": (
            pivot[("Global_Entropy", "3m")] - pivot[("Global_Entropy", "pre")]
    ),
    }).reset_index(drop=True)
    complete = changes.dropna(subset=["Delta_MMSE_3m_minus_0m", "Delta_Global_Entropy_3m_minus_0m"])
    if len(complete) < 3:
        return changes, {"N": len(complete), "Spearman_rho": np.nan, "p_value": np.nan}
    rho, p_value = stats.spearmanr(complete["Delta_Global_Entropy_3m_minus_0m"],
                                   complete["Delta_MMSE_3m_minus_0m"])
    return changes, {"N": len(complete), "Spearman_rho": rho, "p_value": p_value}


def make_change_scatter(changes, spearman_result, out_path):
    complete = changes.dropna(subset=["Delta_MMSE_3m_minus_0m", "Delta_Global_Entropy_3m_minus_0m"])
    figure, axis = plt.subplots(figsize=(5.5, 4.5))
    axis.scatter(complete["Delta_Global_Entropy_3m_minus_0m"], complete["Delta_MMSE_3m_minus_0m"],
                 color="#6A3D9A", edgecolor="white", linewidth=0.5, s=48)
    if len(complete) >= 2 and complete["Delta_Global_Entropy_3m_minus_0m"].nunique() > 1:
        slope, intercept = np.polyfit(complete["Delta_Global_Entropy_3m_minus_0m"],
                                      complete["Delta_MMSE_3m_minus_0m"], 1)
        x_values = np.linspace(complete["Delta_Global_Entropy_3m_minus_0m"].min(),
                               complete["Delta_Global_Entropy_3m_minus_0m"].max(), 100)
        axis.plot(x_values, intercept + slope * x_values, color="black", linewidth=1.2,
                  linestyle="--", label="Linear fit for display")
    axis.axhline(0, color="#999999", linewidth=0.8)
    axis.axvline(0, color="#999999", linewidth=0.8)
    axis.set_xlabel("Change in global entropy (3m − 0m)")
    axis.set_ylabel("Change in MMSE (3m − 0m)")
    rho, p_value = spearman_result["Spearman_rho"], spearman_result["p_value"]
    annotation = "Spearman association unavailable" if not np.isfinite(rho) else (
        f"Spearman ρ = {rho:.3f}\np = {p_value:.3f}; n = {spearman_result['N']}"
    )
    axis.text(0.03, 0.97, annotation, transform=axis.transAxes, ha="left", va="top", fontsize=9)
    if axis.get_legend_handles_labels()[0]:
        axis.legend(frameon=False, fontsize=8, loc="best")
    figure.tight_layout()
    figure.savefig(out_path, dpi=250)
    plt.close(figure)


def rounded(table):
    table = table.copy()
    numeric = table.select_dtypes(include=np.number).columns
    table[numeric] = table[numeric].round(4)
    return table


def run_task_specific_associations(cognitive, task_entropy, output_dir):
    """Secondary task-specific MMSE associations, FDR-adjusted over four tasks."""
    all_models, spearman_rows, all_changes, failures = [], [], [], []
    tasks = sorted(task_entropy["Task"].dropna().astype(str).unique())
    for task in tasks:
        task_composite = task_entropy[task_entropy["Task"].astype(str) == task].copy()
        # Reuse the within/between model with one entropy value per subject x timepoint.
        task_composite = task_composite.rename(columns={"Task_Entropy": "Global_Entropy"})
        data = prepare_model_data(cognitive, task_composite.drop(columns="Task"))
        data["Task"] = task
        fitted_result, status, model_data = fit_lmm(data)
        if fitted_result is None:
            failures.append({"Task": task, "Analysis": "MMSE-task entropy LMM", "Reason": status})
        else:
            fitted, fit_warnings, model_data = fitted_result
            model = model_table(fitted, fit_warnings, model_data)
            model.insert(0, "Task", task)
            all_models.append(model)

        changes, spearman = spearman_change(data)
        changes.insert(0, "Task", task)
        changes = changes.rename(columns={"Delta_Global_Entropy_3m_minus_0m": "Delta_Task_Entropy_3m_minus_0m"})
        all_changes.append(changes)
        spearman_rows.append({"Task": task, **spearman})
        safe_task = task.replace(" ", "_").replace("/", "_")
        make_change_scatter(
            changes.rename(columns={"Delta_Task_Entropy_3m_minus_0m": "Delta_Global_Entropy_3m_minus_0m"}),
            spearman, output_dir / f"mmse_task_entropy_spearman_0m_3m_task_{safe_task}.png"
        )

    model_table_all = pd.concat(all_models, ignore_index=True) if all_models else pd.DataFrame()
    if not model_table_all.empty:
        within = model_table_all["Term"] == "Entropy_within"
        model_table_all["q_FDR_across_tasks"] = np.nan
        model_table_all.loc[within, "q_FDR_across_tasks"] = fdr_bh(
            model_table_all.loc[within, "p_value"].to_numpy()
        )
        rounded(model_table_all).to_csv(output_dir / "mmse_task_entropy_association_lmm.csv", index=False)

    spearman_table = pd.DataFrame(spearman_rows)
    if not spearman_table.empty:
        spearman_table["q_FDR_across_tasks"] = fdr_bh(spearman_table["p_value"].to_numpy())
        rounded(spearman_table).to_csv(output_dir / "mmse_task_entropy_spearman_0m_3m.csv", index=False)
    pd.concat(all_changes, ignore_index=True).to_csv(
        output_dir / "mmse_task_entropy_change_scores_0m_3m.csv", index=False
    )
    task_entropy.to_csv(output_dir / "mmse_task_entropy_composite_long.csv", index=False)
    return pd.DataFrame(failures, columns=["Task", "Analysis", "Reason"])


def main(cognitive_path, entropy_path, output_dir):
    cognitive = read_cognitive(cognitive_path)
    entropy = read_global_entropy(entropy_path)
    task_entropy = read_task_entropy(entropy_path)
    data = prepare_model_data(cognitive, entropy)
    output_dir.mkdir(parents=True, exist_ok=True)
    data.assign(Timepoint=data["timepoint"].map(TIME_LABELS)).to_csv(
        output_dir / "mmse_global_entropy_composite_long.csv", index=False
    )
    fitted_result, status, model_data = fit_lmm(data)
    failures = []
    if fitted_result is None:
        failures.append({"Analysis": "MMSE-entropy within/between LMM", "Reason": status})
    else:
        fitted, fit_warnings, model_data = fitted_result
        rounded(model_table(fitted, fit_warnings, model_data)).to_csv(
            output_dir / "mmse_entropy_association_lmm.csv", index=False
        )
    changes, spearman_result = spearman_change(data)
    changes.to_csv(output_dir / "mmse_entropy_change_scores_0m_3m.csv", index=False)
    pd.DataFrame([spearman_result]).round(4).to_csv(
        output_dir / "mmse_entropy_spearman_0m_3m.csv", index=False
    )
    make_change_scatter(changes, spearman_result, output_dir / "mmse_entropy_spearman_0m_3m.png")
    task_failures = run_task_specific_associations(cognitive, task_entropy, output_dir)
    pd.DataFrame(failures, columns=["Analysis", "Reason"]).to_csv(
        output_dir / "mmse_entropy_association_failures.csv", index=False
    )
    task_failures.to_csv(output_dir / "mmse_task_entropy_association_failures.csv", index=False)
    print(f"Saved results to: {output_dir}")
    print(f"LMM status: {'fitted' if fitted_result is not None else status}")
    print(f"Spearman n = {spearman_result['N']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MMSE-global entropy longitudinal association analysis.")
    parser.add_argument("cognitive_csv", help="Wide cognitive CSV containing MMSE_pre/post/3m")
    parser.add_argument("entropy_long_csv", help="entropy_long.csv from eeg_feature_family_lmm.py")
    parser.add_argument("--output-dir", default="results_mmse_entropy_association",
                        help="Output directory (default: results_mmse_entropy_association)")
    arguments = parser.parse_args()
    try:
        main(arguments.cognitive_csv, arguments.entropy_long_csv, Path(arguments.output_dir))
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
