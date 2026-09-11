# -*- coding: utf-8 -*-
"""
Longitudinal EEG analysis using a linear mixed-effects model.

For every task x ROI x feature, the model is:
    value ~ Post + Month3 + (1 | subject)

Pre is the reference timepoint. Rows with a missing feature value remain NaN
in the wide input and are excluded only from that feature's model; the other
observations from the same subject remain available. The script requires
numpy, pandas, scipy, matplotlib, and statsmodels.

Outputs are written to results_lmm/ in the current working directory.

Usage:
    python eeg_longitudinal_lmm_stats.py wide_feature_table.csv
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

try:
    import statsmodels.api as sm
except ImportError as error:
    raise SystemExit(
        "This script requires statsmodels. Install it in the same Python environment "
        "used to run the script: pip install statsmodels"
    ) from error


ALPHA = 0.05
TASKS = ["3B", "4", "5", "6"]
ROIS = ["Frontal", "Central", "Parietal", "Temporal", "Occipital"]
BANDS = ["delta", "theta", "alpha", "beta"]
TIMEPOINTS = ["pre", "post", "3m"]
CONTRASTS = ["post_minus_pre", "3m_minus_pre", "3m_minus_post"]
FEATURE_FAMILIES = {
    "band_power": [f"{band}_relpower" for band in BANDS],
    "theta_alpha_ratio": ["theta_alpha_ratio"],
    "theta_beta_ratio": ["theta_beta_ratio"],
    "entropy": ["entropy"],
}
SUBJECT_ID_CANDIDATES = ["subject_id", "Subject_ID", "subject", "Subject", "ID", "id"]


def fdr_bh(p_values):
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p_values), np.nan)
    valid = ~np.isnan(p_values)
    if not valid.any():
        return adjusted
    values = p_values[valid]
    order = np.argsort(values)
    ranked = values[order]
    corrected = ranked * len(values) / np.arange(1, len(values) + 1)
    corrected = np.minimum.accumulate(corrected[::-1])[::-1]
    reordered = np.empty(len(values))
    reordered[order] = np.clip(corrected, 0, 1)
    adjusted[valid] = reordered
    return adjusted


def sig_stars(p_value):
    if pd.isna(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < ALPHA:
        return "*"
    return ""


def get_subject_ids(df):
    for column in SUBJECT_ID_CANDIDATES:
        if column in df.columns:
            if df[column].isna().any() or not df[column].is_unique:
                raise ValueError(f"Subject identifier column {column!r} contains missing or duplicate values.")
            return df[column].astype(str), column
    return pd.Series([f"row_{index + 1}" for index in range(len(df))], index=df.index), "generated_row_id"


def build_long_feature(df, subject_ids, columns):
    wide = pd.DataFrame({"subject_id": subject_ids})
    for timepoint, column in columns.items():
        wide[timepoint] = df[column]
    long = wide.melt(id_vars="subject_id", value_vars=TIMEPOINTS,
                     var_name="timepoint", value_name="value").dropna(subset=["value"])
    long["timepoint"] = pd.Categorical(long["timepoint"], categories=TIMEPOINTS, ordered=True)
    return long


def fixed_mean_and_ci(beta, covariance, contrast_vector):
    estimate = float(contrast_vector @ beta)
    standard_error = float(np.sqrt(contrast_vector @ covariance @ contrast_vector))
    return estimate, standard_error, estimate - 1.96 * standard_error, estimate + 1.96 * standard_error


def fit_feature_lmm(long):
    counts = long.groupby("timepoint", observed=False)["value"].count()
    if (counts < 2).any() or long["subject_id"].nunique() < 2:
        return None, "insufficient observations at one or more timepoints"

    exog = pd.DataFrame({
        "Intercept": 1.0,
        "Post": (long["timepoint"] == "post").astype(float).to_numpy(),
        "Month3": (long["timepoint"] == "3m").astype(float).to_numpy(),
    }, index=long.index)
    try:
        fitted = sm.MixedLM(long["value"], exog, groups=long["subject_id"]).fit(reml=True, method="lbfgs")
    except Exception as error:
        return None, f"fit failed: {type(error).__name__}: {error}"
    if not fitted.converged:
        return None, "model did not converge"

    beta = fitted.fe_params.loc[["Intercept", "Post", "Month3"]].to_numpy(dtype=float)
    covariance = fitted.cov_params().loc[["Intercept", "Post", "Month3"],
                                           ["Intercept", "Post", "Month3"]].to_numpy(dtype=float)
    residual_sd = float(np.sqrt(fitted.scale))
    return {
        "beta": beta,
        "covariance": covariance,
        "residual_sd": residual_sd,
        "random_intercept_variance": float(fitted.cov_re.iloc[0, 0]),
        "log_likelihood": float(fitted.llf),
        "aic": float(fitted.aic),
        "bic": float(fitted.bic),
    }, None


def analyze_feature(df, subject_ids, columns):
    long = build_long_feature(df, subject_ids, columns)
    fitted, status = fit_feature_lmm(long)
    if fitted is None:
        return None, status

    beta = fitted["beta"]
    covariance = fitted["covariance"]
    time_beta = beta[1:]
    time_covariance = covariance[1:, 1:]
    wald_chi2 = float(time_beta @ np.linalg.inv(time_covariance) @ time_beta)
    omnibus_p = float(stats.chi2.sf(wald_chi2, df=2))

    mean_vectors = {
        "pre": np.array([1.0, 0.0, 0.0]),
        "post": np.array([1.0, 1.0, 0.0]),
        "3m": np.array([1.0, 0.0, 1.0]),
    }
    model_means = {
        timepoint: fixed_mean_and_ci(beta, covariance, vector)
        for timepoint, vector in mean_vectors.items()
    }
    observed_means = long.groupby("timepoint", observed=False)["value"].mean()
    counts = long.groupby("timepoint", observed=False)["value"].count()
    omnibus = {
        "N_subjects": long["subject_id"].nunique(),
        "N_observations": len(long),
        "N_Pre": int(counts["pre"]),
        "N_Post": int(counts["post"]),
        "N_3m": int(counts["3m"]),
        "Observed_Mean_Pre": observed_means["pre"],
        "Observed_Mean_Post": observed_means["post"],
        "Observed_Mean_3m": observed_means["3m"],
        "Model_Mean_Pre": model_means["pre"][0],
        "Model_Mean_Post": model_means["post"][0],
        "Model_Mean_3m": model_means["3m"][0],
        "Wald_Chi2": wald_chi2,
        "Wald_df": 2,
        "p_value": omnibus_p,
        "Residual_SD": fitted["residual_sd"],
        "Random_Intercept_Variance": fitted["random_intercept_variance"],
        "AIC": fitted["aic"],
        "BIC": fitted["bic"],
    }

    contrast_vectors = {
        "post_minus_pre": np.array([0.0, 1.0, 0.0]),
        "3m_minus_pre": np.array([0.0, 0.0, 1.0]),
        "3m_minus_post": np.array([0.0, -1.0, 1.0]),
    }
    contrasts = []
    for contrast_name, vector in contrast_vectors.items():
        estimate, standard_error, ci_low, ci_high = fixed_mean_and_ci(beta, covariance, vector)
        z_value = estimate / standard_error if standard_error > 0 else np.nan
        p_value = float(2 * stats.norm.sf(abs(z_value))) if not np.isnan(z_value) else np.nan
        contrasts.append({
            "Contrast": contrast_name,
            "Estimate": estimate,
            "SE": standard_error,
            "CI_95_Low": ci_low,
            "CI_95_High": ci_high,
            "z_value": z_value,
            "p_value": p_value,
            "Effect_Size_Type": "Estimate / residual SD",
            "Effect_Size": estimate / fitted["residual_sd"],
        })
    return (omnibus, contrasts, long, model_means), None


def run_all_tasks(df, subject_ids):
    omnibus_by_task, contrasts_by_task, trajectories, failures = {}, {}, [], []
    for task in TASKS:
        task_omnibus, task_contrasts = [], []
        for family_name, measures in FEATURE_FAMILIES.items():
            family_omnibus, family_contrasts = [], []
            for roi in ROIS:
                for measure in measures:
                    base = f"{task}_{roi}_{measure}"
                    columns = {timepoint: f"{base}_{timepoint}" for timepoint in TIMEPOINTS}
                    if not all(column in df.columns for column in columns.values()):
                        failures.append({"Task": task, "ROI": roi, "Measure": measure,
                                         "Reason": "missing one or more required columns"})
                        continue
                    result, status = analyze_feature(df, subject_ids, columns)
                    if result is None:
                        failures.append({"Task": task, "ROI": roi, "Measure": measure, "Reason": status})
                        continue
                    omnibus, contrasts, long, model_means = result
                    metadata = {"Task": task, "Family": family_name, "ROI": roi, "Measure": measure}
                    omnibus.update(metadata)
                    family_omnibus.append(omnibus)
                    for contrast in contrasts:
                        contrast.update(metadata)
                        family_contrasts.append(contrast)
                    trajectories.append((metadata, long, model_means))

            if family_omnibus:
                omnibus_family = pd.DataFrame(family_omnibus)
                omnibus_family["p_adj_FDR"] = fdr_bh(omnibus_family["p_value"].to_numpy())
                omnibus_family["Significant_after_FDR"] = omnibus_family["p_adj_FDR"] < ALPHA
                task_omnibus.append(omnibus_family)
                contrasts_family = pd.DataFrame(family_contrasts)
                contrasts_family["p_adj_FDR"] = fdr_bh(contrasts_family["p_value"].to_numpy())
                contrasts_family["Significant_after_FDR"] = contrasts_family["p_adj_FDR"] < ALPHA
                task_contrasts.append(contrasts_family)

        if task_omnibus:
            omnibus_by_task[task] = pd.concat(task_omnibus, ignore_index=True)
            contrasts_by_task[task] = pd.concat(task_contrasts, ignore_index=True)
        else:
            print(f"WARNING: no fitted models for task {task}")
    return omnibus_by_task, contrasts_by_task, trajectories, pd.DataFrame(failures)


def make_lmm_heatmap(task_df, task, family_name, measures, contrast_name, out_path):
    subset = task_df[(task_df["Family"] == family_name) & (task_df["Contrast"] == contrast_name)]
    if subset.empty:
        return False
    effects = subset.pivot(index="ROI", columns="Measure", values="Effect_Size").reindex(index=ROIS, columns=measures)
    q_values = subset.pivot(index="ROI", columns="Measure", values="p_adj_FDR").reindex(index=ROIS, columns=measures)
    values = effects.to_numpy(dtype=float)
    maximum = np.nanmax(np.abs(values)) if not np.all(np.isnan(values)) else 1.0
    fig, ax = plt.subplots(figsize=(max(5.5, 1.8 * len(measures)), 5))
    image = ax.imshow(values, cmap="RdBu_r", vmin=-maximum, vmax=maximum, aspect="auto")
    ax.set_xticks(range(len(measures)))
    ax.set_xticklabels(measures, rotation=45, ha="right")
    ax.set_yticks(range(len(ROIS)))
    ax.set_yticklabels(ROIS)
    for row in range(len(ROIS)):
        for column in range(len(measures)):
            q_value = q_values.to_numpy()[row, column]
            if pd.notna(q_value) and q_value < ALPHA:
                ax.text(column, row, f"{sig_stars(q_value)}\nq={q_value:.3f}",
                        ha="center", va="center", color="white", fontweight="bold", fontsize=9)
    ax.set_title(f"Task {task} - {family_name} - {contrast_name}\nColor = LMM estimate / residual SD")
    fig.colorbar(image, ax=ax, label="Standardized model estimate", shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def make_lmm_dotplot(task_df, task, family_name, contrast_name, out_path):
    subset = task_df[(task_df["Family"] == family_name) & (task_df["Contrast"] == contrast_name)]
    if subset.empty:
        return False
    subset = subset.set_index("ROI").reindex(ROIS)
    fig, ax = plt.subplots(figsize=(6, 4))
    for position, roi in enumerate(ROIS):
        row = subset.loc[roi]
        if pd.isna(row["Estimate"]):
            continue
        significant = row["p_adj_FDR"] < ALPHA
        color = "#C1272D" if row["Estimate"] > 0 else "#1F77B4"
        ax.errorbar(row["Estimate"], position,
                    xerr=[[row["Estimate"] - row["CI_95_Low"]], [row["CI_95_High"] - row["Estimate"]]],
                    fmt="o", color=color, ecolor=color, capsize=4, markersize=8,
                    markerfacecolor=color if significant else "white", markeredgecolor=color)
        if significant:
            ax.text(row["CI_95_High"], position, f"  {sig_stars(row['p_adj_FDR'])} q={row['p_adj_FDR']:.3f}",
                    va="center", fontsize=9)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_yticks(range(len(ROIS)))
    ax.set_yticklabels(ROIS)
    ax.set_xlabel("LMM estimated difference with 95% CI")
    ax.set_title(f"Task {task} - {family_name} - {contrast_name}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def make_lmm_trajectory(metadata, long, model_means, omnibus_q, out_path):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    x_positions = np.arange(len(TIMEPOINTS))
    pivot = long.pivot(index="subject_id", columns="timepoint", values="value").reindex(columns=TIMEPOINTS)
    ax.plot(x_positions, pivot.to_numpy(dtype=float).T, color="lightgray", linewidth=0.8, alpha=0.7)
    estimates = np.array([model_means[timepoint][0] for timepoint in TIMEPOINTS])
    lows = np.array([model_means[timepoint][2] for timepoint in TIMEPOINTS])
    highs = np.array([model_means[timepoint][3] for timepoint in TIMEPOINTS])
    ax.errorbar(x_positions, estimates, yerr=[estimates - lows, highs - estimates],
                fmt="o-", color="black", capsize=5, linewidth=2, label="LMM estimated mean, 95% CI")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(["Pre", "Post", "3m"])
    ax.set_ylabel(metadata["Measure"])
    ax.set_title(f"Task {metadata['Task']} - {metadata['ROI']} - {metadata['Measure']}\nLMM omnibus q={omnibus_q:.3f}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def rounded_for_export(results):
    results = results.copy()
    numeric = results.select_dtypes(include=np.number).columns
    results[numeric] = results[numeric].round(4)
    return results


def main(csv_path):
    df = pd.read_csv(csv_path)
    subject_ids, subject_id_source = get_subject_ids(df)
    output_dir = Path("results_lmm")
    output_dir.mkdir(exist_ok=True)
    print(f"Loaded {csv_path}: {len(df)} rows, {len(df.columns)} columns; subject ID: {subject_id_source}")
    omnibus_by_task, contrasts_by_task, trajectories, failures = run_all_tasks(df, subject_ids)
    if not omnibus_by_task:
        raise ValueError("No LMMs could be fitted.")

    omnibus_summary = pd.concat(omnibus_by_task.values(), ignore_index=True)
    contrasts_summary = pd.concat(contrasts_by_task.values(), ignore_index=True)
    rounded_for_export(omnibus_summary).to_csv(output_dir / "eeg_lmm_omnibus_summary.csv", index=False)
    rounded_for_export(contrasts_summary).to_csv(output_dir / "eeg_lmm_contrasts_summary.csv", index=False)
    failures.to_csv(output_dir / "eeg_lmm_failures.csv", index=False)
    for task, task_omnibus in omnibus_by_task.items():
        rounded_for_export(task_omnibus).to_csv(output_dir / f"eeg_lmm_omnibus_task_{task}.csv", index=False)
        task_contrasts = contrasts_by_task[task]
        rounded_for_export(task_contrasts).to_csv(output_dir / f"eeg_lmm_contrasts_task_{task}.csv", index=False)
        for family_name, measures in FEATURE_FAMILIES.items():
            for contrast_name in CONTRASTS:
                plot_path = output_dir / f"eeg_lmm_task_{task}_{family_name}_{contrast_name}.png"
                if len(measures) > 1:
                    make_lmm_heatmap(task_contrasts, task, family_name, measures, contrast_name, plot_path)
                else:
                    make_lmm_dotplot(task_contrasts, task, family_name, contrast_name, plot_path)

    significant_keys = set(
        tuple(row) for row in omnibus_summary.loc[omnibus_summary["Significant_after_FDR"],
                                                 ["Task", "ROI", "Measure"]].to_numpy()
    )
    for metadata, long, model_means in trajectories:
        key = (metadata["Task"], metadata["ROI"], metadata["Measure"])
        if key not in significant_keys:
            continue
        omnibus_q = omnibus_summary.loc[
            (omnibus_summary["Task"] == key[0]) & (omnibus_summary["ROI"] == key[1])
            & (omnibus_summary["Measure"] == key[2]), "p_adj_FDR"
        ].iloc[0]
        make_lmm_trajectory(metadata, long, model_means, omnibus_q,
                            output_dir / f"eeg_lmm_trajectory_task_{key[0]}_{key[1]}_{key[2]}.png")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python eeg_longitudinal_lmm_stats.py wide_feature_table.csv")
        sys.exit(1)
    main(sys.argv[1])
