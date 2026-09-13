# -*- coding: utf-8 -*-
"""Longitudinal analysis for the BrainTrain cognitive-test battery.

Input is one wide CSV row per subject with columns:
    subject_id, MMSE_pre, MMSE_post, MMSE_3m,
    TMTA_pre, TMTA_post, TMTA_3m,
    TMTB_pre, TMTB_post, TMTB_3m,
    DSF_pre, DSF_post, DSF_3m, DSB_pre, DSB_post, DSB_3m

Primary analyses
------------------
* MMSE, TMT-A, Digit Span Forward and Digit Span Backward:
      score ~ timepoint + (1 | subject)
  TMT-A is analysed on log(seconds).  Timepoint is categorical, with 0m as
  the reference, so no linear change over time is assumed.
* TMT-B completion: logistic GEE clustered by subject.  A value of 300 means
  the participant did not complete within the protocol limit and is therefore
  analysed as non-completion, not as an observed completion time.
* TMT-B completion time: a secondary log-linear mixed model among completed
  assessments only (TMT-B < 300).  It must not be interpreted as the outcome
  for the entire cohort.

Missing values remain NaN.  Each model uses all observed rows for its own
outcome; no mean/median imputation or complete-case deletion is performed.

FDR strategy
------------
Primary omnibus time tests are adjusted across the five primary outcomes.
Primary planned contrasts (1m-0m, 3m-0m, 3m-1m) are adjusted across all
15 primary outcome-by-contrast tests.  Secondary TMT-B completion-time tests
are adjusted only within that clearly labelled secondary analysis.

Usage
-----
    python3 cognitive_longitudinal_lmm.py cogtest_013m.csv

Outputs are written to results_cognitive_lmm/ beside the current working
directory unless --output-dir is supplied.
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
    from matplotlib.lines import Line2D
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


ALPHA = 0.05
TIMEPOINTS = ["pre", "post", "3m"]
TIME_LABELS = {"pre": "0m", "post": "1m", "3m": "3m"}
TIME_COLORS = {"pre": "#377EB8", "post": "#FF7F00", "3m": "#984EA3"}
CONTRASTS = {
    "1m_minus_0m": np.array([0.0, 1.0, 0.0]),
    "3m_minus_0m": np.array([0.0, 0.0, 1.0]),
    "3m_minus_1m": np.array([0.0, -1.0, 1.0]),
}
MODEL_MEAN_VECTORS = {
    "pre": np.array([1.0, 0.0, 0.0]),
    "post": np.array([1.0, 1.0, 0.0]),
    "3m": np.array([1.0, 0.0, 1.0]),
}
CONTINUOUS_OUTCOMES = {
    "MMSE": {
        "columns": {"pre": "MMSE_pre", "post": "MMSE_post", "3m": "MMSE_3m"},
        "label": "MMSE score",
        "lower_is_better": False,
        "log_transform": False,
        "analysis_role": "Primary",
        "y_max": 30,
    },
    "TMT-A": {
        "columns": {"pre": "TMTA_pre", "post": "TMTA_post", "3m": "TMTA_3m"},
        "label": "TMT-A completion time (s)",
        "lower_is_better": True,
        "log_transform": True,
        "analysis_role": "Primary",
    },
    "Digit Span Forward": {
        "columns": {"pre": "DSF_pre", "post": "DSF_post", "3m": "DSF_3m"},
        "label": "Digit Span Forward score",
        "lower_is_better": False,
        "log_transform": False,
        "analysis_role": "Primary",
    },
    "Digit Span Backward": {
        "columns": {"pre": "DSB_pre", "post": "DSB_post", "3m": "DSB_3m"},
        "label": "Digit Span Backward score",
        "lower_is_better": False,
        "log_transform": False,
        "analysis_role": "Primary",
    },
}
TMTB_COLUMNS = {"pre": "TMTB_pre", "post": "TMTB_post", "3m": "TMTB_3m"}
REQUIRED_COLUMNS = ["subject_id"] + [
    column for outcome in CONTINUOUS_OUTCOMES.values() for column in outcome["columns"].values()
] + list(TMTB_COLUMNS.values())


def fdr_bh(p_values):
    """Benjamini-Hochberg adjusted p-values, preserving missing values."""
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


def sig_stars(q_value):
    if pd.isna(q_value):
        return ""
    if q_value < 0.001:
        return "***"
    if q_value < 0.01:
        return "**"
    if q_value < ALPHA:
        return "*"
    return ""


def time_legend_label(timepoint):
    return {
        "pre": "0m (baseline)",
        "post": "1m (1 month)",
        "3m": "3m (3 months)",
    }[timepoint]


def clean_input(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")].copy()
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    if df["subject_id"].isna().any() or not df["subject_id"].is_unique:
        raise ValueError("subject_id must contain one non-missing, unique ID per row.")
    df["subject_id"] = df["subject_id"].astype(str).str.strip()
    numeric_columns = [column for column in REQUIRED_COLUMNS if column != "subject_id"]
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")
    return df


def build_long(df, columns, value_name="value"):
    wide = df[["subject_id"]].copy()
    for timepoint, column in columns.items():
        wide[timepoint] = df[column]
    long = wide.melt(id_vars="subject_id", value_vars=TIMEPOINTS,
                     var_name="timepoint", value_name=value_name)
    long["timepoint"] = pd.Categorical(long["timepoint"], categories=TIMEPOINTS, ordered=True)
    return long


def design_matrix(long):
    return pd.DataFrame({
        "Intercept": 1.0,
        "Month1": (long["timepoint"] == "post").astype(float).to_numpy(),
        "Month3": (long["timepoint"] == "3m").astype(float).to_numpy(),
    }, index=long.index)


def estimate_and_ci(beta, covariance, vector):
    estimate = float(vector @ beta)
    variance = float(vector @ covariance @ vector)
    standard_error = float(np.sqrt(max(variance, 0.0)))
    return estimate, standard_error, estimate - 1.96 * standard_error, estimate + 1.96 * standard_error


def fit_lmm(long, model_value_column="model_value"):
    counts = long.groupby("timepoint", observed=False)[model_value_column].count()
    if (counts < 2).any() or long["subject_id"].nunique() < 2:
        return None, "insufficient observations at one or more timepoints"

    exog = design_matrix(long)
    attempts, captured_messages = [], []
    for method in ("lbfgs", "powell"):
        try:
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                fitted = sm.MixedLM(long[model_value_column], exog, groups=long["subject_id"]).fit(
                    reml=True, method=method
                )
            if fitted.converged:
                captured_messages = [str(item.message) for item in captured]
                beta = fitted.fe_params.loc[["Intercept", "Month1", "Month3"]].to_numpy(dtype=float)
                covariance = fitted.cov_params().loc[
                    ["Intercept", "Month1", "Month3"], ["Intercept", "Month1", "Month3"]
                ].to_numpy(dtype=float)
                return {
                    "beta": beta,
                    "covariance": covariance,
                    "residual_sd": float(np.sqrt(fitted.scale)),
                    "random_intercept_variance": float(fitted.cov_re.iloc[0, 0]),
                    "aic": float(fitted.aic),
                    "bic": float(fitted.bic),
                    "fit_method": method,
                    "fit_warning": " | ".join(captured_messages),
                }, None
            attempts.append(f"{method}: did not converge")
        except Exception as error:
            attempts.append(f"{method}: {type(error).__name__}: {error}")
    return None, "fit failed: " + " | ".join(attempts)


def analyse_continuous(df, outcome, config, secondary=False):
    long = build_long(df, config["columns"]).dropna(subset=["value"]).copy()
    if config["log_transform"]:
        invalid = long["value"] <= 0
        if invalid.any():
            return None, f"non-positive values cannot be log-transformed ({int(invalid.sum())} rows)"
        long["model_value"] = np.log(long["value"])
    else:
        long["model_value"] = long["value"]

    fitted, status = fit_lmm(long)
    if fitted is None:
        return None, status

    beta, covariance = fitted["beta"], fitted["covariance"]
    time_beta, time_covariance = beta[1:], covariance[1:, 1:]
    omnibus_chi2 = float(time_beta @ np.linalg.pinv(time_covariance) @ time_beta)
    omnibus_p = float(stats.chi2.sf(omnibus_chi2, df=2))
    counts = long.groupby("timepoint", observed=False)["value"].count()
    raw_means = long.groupby("timepoint", observed=False)["value"].mean()

    model_means = {}
    for timepoint, vector in MODEL_MEAN_VECTORS.items():
        estimate, se, ci_low, ci_high = estimate_and_ci(beta, covariance, vector)
        if config["log_transform"]:
            model_means[timepoint] = (np.exp(estimate), np.exp(ci_low), np.exp(ci_high))
        else:
            model_means[timepoint] = (estimate, ci_low, ci_high)

    omnibus = {
        "Outcome": outcome,
        "Analysis": "Secondary TMT-B time among completers" if secondary else "Primary LMM",
        "Model": "log(seconds) ~ timepoint + (1 | subject)" if config["log_transform"] else "score ~ timepoint + (1 | subject)",
        "N_subjects": int(long["subject_id"].nunique()),
        "N_observations": int(len(long)),
        "N_0m": int(counts["pre"]),
        "N_1m": int(counts["post"]),
        "N_3m": int(counts["3m"]),
        "Observed_Mean_0m": raw_means["pre"],
        "Observed_Mean_1m": raw_means["post"],
        "Observed_Mean_3m": raw_means["3m"],
        "Model_Mean_0m": model_means["pre"][0],
        "Model_Mean_1m": model_means["post"][0],
        "Model_Mean_3m": model_means["3m"][0],
        "Wald_Chi2": omnibus_chi2,
        "Wald_df": 2,
        "p_value": omnibus_p,
        "Residual_SD": fitted["residual_sd"],
        "Random_Intercept_Variance": fitted["random_intercept_variance"],
        "AIC": fitted["aic"],
        "BIC": fitted["bic"],
        "Fit_Method": fitted["fit_method"],
        "Fit_Warning": fitted["fit_warning"],
    }
    contrasts = []
    for name, vector in CONTRASTS.items():
        estimate, se, ci_low, ci_high = estimate_and_ci(beta, covariance, vector)
        z_value = estimate / se if se > 0 else np.nan
        p_value = float(2 * stats.norm.sf(abs(z_value))) if np.isfinite(z_value) else np.nan
        row = {
            "Outcome": outcome,
            "Analysis": omnibus["Analysis"],
            "Contrast": name,
            "Estimate": estimate,
            "SE": se,
            "CI_95_Low": ci_low,
            "CI_95_High": ci_high,
            "z_value": z_value,
            "p_value": p_value,
            "Lower_is_better": config["lower_is_better"],
        }
        if config["log_transform"]:
            row.update({
                "Estimate_scale": "log ratio of seconds",
                "Ratio_of_times": np.exp(estimate),
                "Ratio_CI_95_Low": np.exp(ci_low),
                "Ratio_CI_95_High": np.exp(ci_high),
                "Percent_change_in_time": (np.exp(estimate) - 1.0) * 100,
            })
        else:
            row.update({
                "Estimate_scale": "score difference",
                "Standardized_estimate": estimate / fitted["residual_sd"],
            })
        contrasts.append(row)
    return {"omnibus": omnibus, "contrasts": contrasts, "long": long, "model_means": model_means}, None


def analyse_tmtb_completion(df):
    long = build_long(df, TMTB_COLUMNS).dropna(subset=["value"]).copy()
    invalid = (long["value"] <= 0) | (long["value"] > 300)
    if invalid.any():
        return None, "TMT-B values must be in (0, 300] when observed"
    long["completed"] = (long["value"] < 300).astype(int)
    counts = long.groupby("timepoint", observed=False)["completed"].agg(["count", "sum"])
    if (counts["count"] < 2).any() or long["completed"].nunique() < 2:
        return None, "insufficient observations or no variation in TMT-B completion"
    exog = design_matrix(long)
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            fitted = sm.GEE(
                long["completed"], exog, groups=long["subject_id"],
                family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()
            ).fit()
    except Exception as error:
        return None, f"fit failed: {type(error).__name__}: {error}"

    beta = fitted.params.loc[["Intercept", "Month1", "Month3"]].to_numpy(dtype=float)
    covariance = fitted.cov_params().loc[
        ["Intercept", "Month1", "Month3"], ["Intercept", "Month1", "Month3"]
    ].to_numpy(dtype=float)
    time_beta, time_covariance = beta[1:], covariance[1:, 1:]
    omnibus_chi2 = float(time_beta @ np.linalg.pinv(time_covariance) @ time_beta)
    omnibus_p = float(stats.chi2.sf(omnibus_chi2, df=2))

    model_probabilities = {}
    for timepoint, vector in MODEL_MEAN_VECTORS.items():
        linear, _, ci_low, ci_high = estimate_and_ci(beta, covariance, vector)
        model_probabilities[timepoint] = (
            float(sm.families.links.Logit().inverse(linear)),
            float(sm.families.links.Logit().inverse(ci_low)),
            float(sm.families.links.Logit().inverse(ci_high)),
        )

    omnibus = {
        "Outcome": "TMT-B completion before 300 s",
        "Analysis": "Primary logistic GEE",
        "Model": "completion ~ timepoint; subject-clustered logistic GEE",
        "N_subjects": int(long["subject_id"].nunique()),
        "N_observations": int(len(long)),
        "N_0m": int(counts.loc["pre", "count"]),
        "N_1m": int(counts.loc["post", "count"]),
        "N_3m": int(counts.loc["3m", "count"]),
        "Completed_0m": int(counts.loc["pre", "sum"]),
        "Completed_1m": int(counts.loc["post", "sum"]),
        "Completed_3m": int(counts.loc["3m", "sum"]),
        "Observed_Proportion_0m": counts.loc["pre", "sum"] / counts.loc["pre", "count"],
        "Observed_Proportion_1m": counts.loc["post", "sum"] / counts.loc["post", "count"],
        "Observed_Proportion_3m": counts.loc["3m", "sum"] / counts.loc["3m", "count"],
        "Model_Probability_0m": model_probabilities["pre"][0],
        "Model_Probability_1m": model_probabilities["post"][0],
        "Model_Probability_3m": model_probabilities["3m"][0],
        "Wald_Chi2": omnibus_chi2,
        "Wald_df": 2,
        "p_value": omnibus_p,
        "Working_correlation": float(fitted.cov_struct.dep_params),
        "Fit_Warning": " | ".join(str(item.message) for item in captured),
    }
    contrasts = []
    for name, vector in CONTRASTS.items():
        estimate, se, ci_low, ci_high = estimate_and_ci(beta, covariance, vector)
        z_value = estimate / se if se > 0 else np.nan
        p_value = float(2 * stats.norm.sf(abs(z_value))) if np.isfinite(z_value) else np.nan
        contrasts.append({
            "Outcome": "TMT-B completion before 300 s",
            "Analysis": "Primary logistic GEE",
            "Contrast": name,
            "Estimate_log_OR": estimate,
            "SE": se,
            "CI_95_Low_log_OR": ci_low,
            "CI_95_High_log_OR": ci_high,
            "Odds_Ratio": np.exp(estimate),
            "OR_CI_95_Low": np.exp(ci_low),
            "OR_CI_95_High": np.exp(ci_high),
            "z_value": z_value,
            "p_value": p_value,
        })
    return {"omnibus": omnibus, "contrasts": contrasts, "long": long,
            "model_probabilities": model_probabilities}, None


def descriptive_table(df):
    rows = []
    outcomes = {name: config["columns"] for name, config in CONTINUOUS_OUTCOMES.items()}
    outcomes["TMT-B recorded value"] = TMTB_COLUMNS
    for outcome, columns in outcomes.items():
        for timepoint, column in columns.items():
            values = df[column].dropna()
            rows.append({
                "Outcome": outcome,
                "Timepoint": TIME_LABELS[timepoint],
                "N": int(values.size),
                "Mean": values.mean(),
                "SD": values.std(ddof=1),
                "Median": values.median(),
                "IQR": values.quantile(0.75) - values.quantile(0.25),
                "Min": values.min(),
                "Max": values.max(),
            })
    return pd.DataFrame(rows)


def missingness_table(df):
    rows = []
    outcomes = {name: config["columns"] for name, config in CONTINUOUS_OUTCOMES.items()}
    outcomes["TMT-B"] = TMTB_COLUMNS
    for outcome, columns in outcomes.items():
        for timepoint, column in columns.items():
            n_observed = int(df[column].notna().sum())
            rows.append({
                "Outcome": outcome,
                "Timepoint": TIME_LABELS[timepoint],
                "N_total": len(df),
                "N_observed": n_observed,
                "N_missing": int(df[column].isna().sum()),
            })
    return pd.DataFrame(rows)


def tmtb_status_table(df):
    long = build_long(df, TMTB_COLUMNS)
    rows = []
    for timepoint in TIMEPOINTS:
        values = long.loc[long["timepoint"] == timepoint, "value"]
        observed = values.dropna()
        rows.append({
            "Timepoint": TIME_LABELS[timepoint],
            "N_total": len(values),
            "N_missing": int(values.isna().sum()),
            "N_assessed": int(observed.size),
            "N_completed_under_300s": int((observed < 300).sum()),
            "N_not_completed_300s": int((observed == 300).sum()),
            "Completion_proportion": (observed < 300).mean() if observed.size else np.nan,
        })
    return pd.DataFrame(rows)


def format_mean_sd(values):
    values = values.dropna()
    if values.empty:
        return "NA"
    return f"{values.mean():.2f} ({values.std(ddof=1):.2f})"


def format_q(value):
    if pd.isna(value):
        return "NA"
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def format_effect(row):
    if "Odds_Ratio" in row and pd.notna(row["Odds_Ratio"]):
        return f"OR {row['Odds_Ratio']:.2f} ({row['OR_CI_95_Low']:.2f}, {row['OR_CI_95_High']:.2f})"
    if "Ratio_of_times" in row and pd.notna(row["Ratio_of_times"]):
        return f"Ratio {row['Ratio_of_times']:.2f} ({row['Ratio_CI_95_Low']:.2f}, {row['Ratio_CI_95_High']:.2f})"
    return f"Δ {row['Estimate']:.2f} ({row['CI_95_Low']:.2f}, {row['CI_95_High']:.2f})"


def build_main_table(df, primary_omnibus, primary_contrasts):
    """Create a compact, manuscript-oriented table for the five primary outcomes."""
    primary_order = list(CONTINUOUS_OUTCOMES) + ["TMT-B completion before 300 s"]
    rows = []
    for outcome in primary_order:
        omnibus_row = primary_omnibus.loc[primary_omnibus["Outcome"] == outcome]
        if omnibus_row.empty:
            continue
        omnibus_row = omnibus_row.iloc[0]
        row = {
            "Outcome": outcome,
            "Primary_analysis": omnibus_row["Analysis"],
            "0m": "",
            "1m": "",
            "3m": "",
            "Omnibus_q_FDR": format_q(omnibus_row["q_FDR"]),
        }
        if outcome == "TMT-B completion before 300 s":
            for timepoint, column_name in TMTB_COLUMNS.items():
                assessed = df[column_name].notna().sum()
                completed = (df[column_name] < 300).sum()
                row[TIME_LABELS[timepoint]] = f"{completed}/{assessed} ({completed / assessed * 100:.1f}%)"
            row["Effect_measure"] = "Odds ratio for completion"
        else:
            config = CONTINUOUS_OUTCOMES[outcome]
            for timepoint, column_name in config["columns"].items():
                row[TIME_LABELS[timepoint]] = format_mean_sd(df[column_name])
            row["Effect_measure"] = "Ratio of completion times" if config["log_transform"] else "Mean score difference"
        for contrast in CONTRASTS:
            contrast_row = primary_contrasts.loc[
                (primary_contrasts["Outcome"] == outcome) & (primary_contrasts["Contrast"] == contrast)
            ]
            contrast_label = contrast.replace("_minus_", " vs ")
            if contrast_row.empty:
                row[f"{contrast_label}_effect_95CI"] = "NA"
                row[f"{contrast_label}_q_FDR"] = "NA"
            else:
                contrast_row = contrast_row.iloc[0]
                row[f"{contrast_label}_effect_95CI"] = format_effect(contrast_row)
                row[f"{contrast_label}_q_FDR"] = format_q(contrast_row["q_FDR"])
        rows.append(row)
    return pd.DataFrame(rows)


def add_fdr(results):
    if not results:
        return pd.DataFrame()
    frame = pd.DataFrame(results)
    frame["q_FDR"] = fdr_bh(frame["p_value"].to_numpy())
    frame["Significant_after_FDR"] = frame["q_FDR"] < ALPHA
    return frame


def make_continuous_trajectory(outcome, config, long, model_means, omnibus_q, contrasts, out_path, axis=None):
    created_figure = axis is None
    if created_figure:
        figure, axis = plt.subplots(figsize=(5.7, 4.4))
    else:
        figure = axis.figure
    x = np.arange(3)
    pivot = long.pivot(index="subject_id", columns="timepoint", values="value").reindex(columns=TIMEPOINTS)
    values = pivot.to_numpy(dtype=float)
    axis.plot(x, values.T, color="#B8B8B8", linewidth=0.8, alpha=0.65, zorder=1)
    for position in range(3):
        observed = values[:, position]
        axis.scatter(np.repeat(x[position], np.isfinite(observed).sum()), observed[np.isfinite(observed)],
                     color=TIME_COLORS[TIMEPOINTS[position]], alpha=0.8, s=22, zorder=2,
                     label=time_legend_label(TIMEPOINTS[position]))
    estimates = np.array([model_means[timepoint][0] for timepoint in TIMEPOINTS])
    lows = np.array([model_means[timepoint][1] for timepoint in TIMEPOINTS])
    highs = np.array([model_means[timepoint][2] for timepoint in TIMEPOINTS])
    axis.errorbar(x, estimates, yerr=[estimates - lows, highs - estimates], fmt="o-", color="black",
                  linewidth=2, capsize=4, markersize=5, zorder=4, label="Model mean, 95% CI")
    y_min, y_max = np.nanmin(np.r_[values.ravel(), lows]), np.nanmax(np.r_[values.ravel(), highs])
    span = max(y_max - y_min, 1e-6)
    significant = contrasts.loc[contrasts["q_FDR"] < ALPHA]
    bracket_x = {"1m_minus_0m": (0, 1), "3m_minus_0m": (0, 2), "3m_minus_1m": (1, 2)}
    if "y_max" in config:
        # Keep 30 as MMSE's highest valid tick. The narrow blank margin above it is
        # purely for statistical brackets and is not an extension of the score scale.
        valid_max = config["y_max"]
        bracket_start = valid_max + 0.65
        bracket_step = 0.65
        for level, (_, row) in enumerate(significant.iterrows()):
            left, right = bracket_x[row["Contrast"]]
            height = bracket_start + bracket_step * level
            if row["Contrast"] == "3m_minus_0m":
                height += 0.20
            axis.plot([left, left, right, right], [height - 0.06, height, height, height - 0.06],
                      color="black", linewidth=1)
            axis.text((left + right) / 2, height + 0.04, sig_stars(row["q_FDR"]), ha="center", va="bottom")
        bottom = max(0, y_min - span * 0.05)
        # Leave enough in-axis room between the bracket and the title while
        # retaining 30 as the highest displayed MMSE tick.
        top = valid_max + (1.75 + 0.80 * max(len(significant) - 1, 0) if not significant.empty else 0.25)
        axis.set_ylim(bottom=bottom, top=top)
        start_tick = int(np.floor(bottom / 2.0) * 2)
        axis.set_yticks(np.arange(start_tick, valid_max + 1, 2))
        annotation_y, annotation_va = 0.02, "bottom"
    else:
        for level, (_, row) in enumerate(significant.iterrows()):
            left, right = bracket_x[row["Contrast"]]
            height = y_max + span * (0.12 + 0.12 * level)
            cap = span * 0.025
            axis.plot([left, left, right, right], [height - cap, height, height, height - cap], color="black", linewidth=1)
            axis.text((left + right) / 2, height + span * 0.015, sig_stars(row["q_FDR"]), ha="center", va="bottom")
        if not significant.empty:
            axis.set_ylim(top=y_max + span * (0.18 + 0.12 * len(significant)))
        annotation_y, annotation_va = 0.98, "top"
    axis.set_xticks(x)
    axis.set_xticklabels(["0m", "1m", "3m"])
    axis.set_ylabel(config["label"])
    axis.set_title(outcome)
    axis.text(0.02, annotation_y, f"Time effect: q_FDR = {omnibus_q:.3f}", transform=axis.transAxes,
              ha="left", va=annotation_va, fontsize=8)
    for tick, timepoint in zip(axis.get_xticklabels(), TIMEPOINTS):
        tick.set_color(TIME_COLORS[timepoint])
    if created_figure:
        axis.legend(fontsize=8, loc="best")
        figure.tight_layout()
        figure.savefig(out_path, dpi=200)
        plt.close(figure)


def make_tmtb_completion_plot(result, omnibus_q, contrasts, out_path=None, axis=None):
    long, model_probabilities = result["long"], result["model_probabilities"]
    counts = long.groupby("timepoint", observed=False)["completed"].agg(["count", "sum"])
    observed = (counts["sum"] / counts["count"]).to_numpy(dtype=float)
    x = np.arange(3)
    probabilities = np.array([model_probabilities[timepoint][0] for timepoint in TIMEPOINTS])
    lows = np.array([model_probabilities[timepoint][1] for timepoint in TIMEPOINTS])
    highs = np.array([model_probabilities[timepoint][2] for timepoint in TIMEPOINTS])
    created_figure = axis is None
    if created_figure:
        figure, axis = plt.subplots(figsize=(5.7, 4.4))
    else:
        figure = axis.figure
    for position, timepoint in enumerate(TIMEPOINTS):
        axis.scatter(x[position], observed[position], color=TIME_COLORS[timepoint], s=40, zorder=2,
                     label=time_legend_label(timepoint))
    axis.errorbar(x, probabilities, yerr=[probabilities - lows, highs - probabilities], fmt="o-", color="black",
                  linewidth=2, capsize=4, markersize=5, zorder=3, label="GEE probability, 95% CI")
    significant = contrasts.loc[contrasts["q_FDR"] < ALPHA]
    bracket_x = {"1m_minus_0m": (0, 1), "3m_minus_0m": (0, 2), "3m_minus_1m": (1, 2)}
    for level, (_, row) in enumerate(significant.iterrows()):
        left, right = bracket_x[row["Contrast"]]
        height = min(0.97, 0.86 + 0.07 * level)
        axis.plot([left, left, right, right], [height - 0.02, height, height, height - 0.02], color="black", linewidth=1)
        axis.text((left + right) / 2, min(0.99, height + 0.015), sig_stars(row["q_FDR"]), ha="center", va="bottom")
    axis.set_xticks(x)
    axis.set_xticklabels(["0m", "1m", "3m"])
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Completion proportion before 300 s")
    axis.set_title("TMT-B completion")
    axis.text(0.02, 0.98, f"Time effect: q_FDR = {omnibus_q:.3f}", transform=axis.transAxes,
              ha="left", va="top", fontsize=8)
    for tick, timepoint in zip(axis.get_xticklabels(), TIMEPOINTS):
        tick.set_color(TIME_COLORS[timepoint])
    if created_figure:
        axis.legend(fontsize=8, loc="lower right")
        figure.tight_layout()
        figure.savefig(out_path, dpi=200)
        plt.close(figure)


def rounded(frame):
    frame = frame.copy()
    numeric = frame.select_dtypes(include=np.number).columns
    frame[numeric] = frame[numeric].round(4)
    return frame


def make_cognitive_main_figure(continuous_for_plot, completion_result, primary_omnibus,
                               primary_contrasts, secondary_omnibus, secondary_contrasts, out_path):
    """Create the 3-row x 2-column manuscript overview requested for cognition."""
    figure, axes = plt.subplots(3, 2, figsize=(11.5, 14))
    axes = axes.ravel()
    omnibus_q = primary_omnibus.set_index("Outcome")["q_FDR"]
    ordered_continuous = ["MMSE", "TMT-A", "Digit Span Forward", "Digit Span Backward"]

    for outcome, axis in zip(ordered_continuous[:2], axes[:2]):
        if outcome not in continuous_for_plot:
            axis.text(0.5, 0.5, "Model unavailable", ha="center", va="center")
            axis.axis("off")
            continue
        result, config = continuous_for_plot[outcome]
        make_continuous_trajectory(
            outcome, config, result["long"], result["model_means"], omnibus_q.loc[outcome],
            primary_contrasts[primary_contrasts["Outcome"] == outcome], None, axis=axis
        )

    completion_axis, time_axis = axes[2], axes[3]
    if completion_result is None:
        completion_axis.text(0.5, 0.5, "Model unavailable", ha="center", va="center")
        completion_axis.axis("off")
    else:
        make_tmtb_completion_plot(
            completion_result, omnibus_q.loc["TMT-B completion before 300 s"],
            primary_contrasts[primary_contrasts["Outcome"] == "TMT-B completion before 300 s"],
            axis=completion_axis
        )

    tmtb_time_name = "TMT-B time among completers"
    if tmtb_time_name not in continuous_for_plot:
        time_axis.text(0.5, 0.5, "Secondary model unavailable", ha="center", va="center")
        time_axis.axis("off")
    else:
        result, config = continuous_for_plot[tmtb_time_name]
        make_continuous_trajectory(
            tmtb_time_name, config, result["long"], result["model_means"],
            secondary_omnibus.loc[0, "q_FDR"],
            secondary_contrasts[secondary_contrasts["Outcome"] == tmtb_time_name], None, axis=time_axis
        )

    for outcome, axis in zip(ordered_continuous[2:], axes[4:]):
        if outcome not in continuous_for_plot:
            axis.text(0.5, 0.5, "Model unavailable", ha="center", va="center")
            axis.axis("off")
            continue
        result, config = continuous_for_plot[outcome]
        make_continuous_trajectory(
            outcome, config, result["long"], result["model_means"], omnibus_q.loc[outcome],
            primary_contrasts[primary_contrasts["Outcome"] == outcome], None, axis=axis
        )

    for label, axis in zip("ABCDEF", axes):
        axis.text(-0.14, 1.10, label, transform=axis.transAxes, fontsize=14,
                  fontweight="bold", ha="left", va="top")

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=TIME_COLORS["pre"], markersize=7,
               label=time_legend_label("pre")),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=TIME_COLORS["post"], markersize=7,
               label=time_legend_label("post")),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=TIME_COLORS["3m"], markersize=7,
               label=time_legend_label("3m")),
        Line2D([0], [0], color="#B8B8B8", linewidth=1, label="Individual trajectory"),
        Line2D([0], [0], color="black", marker="o", linewidth=2, markersize=5, label="Model mean, 95% CI"),
    ]
    figure.legend(handles=legend_handles, loc="lower center", ncol=5, frameon=False, fontsize=9,
                  bbox_to_anchor=(0.5, 0.015))
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    figure.savefig(out_path, dpi=250)
    plt.close(figure)


def main(csv_path, output_dir):
    df = clean_input(csv_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loaded {csv_path}: {len(df)} rows, {len(df.columns)} columns; subject ID: subject_id")

    rounded(descriptive_table(df)).to_csv(output_dir / "cognitive_descriptives.csv", index=False)
    missingness_table(df).to_csv(output_dir / "cognitive_missingness_qc.csv", index=False)
    rounded(tmtb_status_table(df)).to_csv(output_dir / "tmtb_completion_status.csv", index=False)

    failures, primary_results, continuous_for_plot = [], [], {}
    for outcome, config in CONTINUOUS_OUTCOMES.items():
        result, status = analyse_continuous(df, outcome, config)
        if result is None:
            failures.append({"Outcome": outcome, "Analysis": "Primary LMM", "Reason": status})
            continue
        primary_results.append(result)
        continuous_for_plot[outcome] = (result, config)

    completion_result, status = analyse_tmtb_completion(df)
    if completion_result is None:
        failures.append({"Outcome": "TMT-B completion", "Analysis": "Primary logistic GEE", "Reason": status})
    else:
        primary_results.append(completion_result)

    primary_omnibus = add_fdr([result["omnibus"] for result in primary_results])
    primary_contrasts = add_fdr([row for result in primary_results for row in result["contrasts"]])
    rounded(primary_omnibus).to_csv(output_dir / "cognitive_primary_omnibus.csv", index=False)
    rounded(primary_contrasts).to_csv(output_dir / "cognitive_primary_contrasts.csv", index=False)
    build_main_table(df, primary_omnibus, primary_contrasts).to_csv(
        output_dir / "cognitive_main_table.csv", index=False
    )

    # Secondary time model only among TMT-B assessments completed before 300 seconds.
    tmtb_completed = df.copy()
    for column in TMTB_COLUMNS.values():
        tmtb_completed.loc[tmtb_completed[column] >= 300, column] = np.nan
    tmtb_config = {
        "columns": TMTB_COLUMNS,
        "label": "TMT-B completion time among completers (s)",
        "lower_is_better": True,
        "log_transform": True,
    }
    secondary_omnibus, secondary_contrasts = pd.DataFrame(), pd.DataFrame()
    secondary_result, status = analyse_continuous(tmtb_completed, "TMT-B time among completers", tmtb_config, secondary=True)
    if secondary_result is None:
        failures.append({"Outcome": "TMT-B time among completers", "Analysis": "Secondary LMM", "Reason": status})
    else:
        secondary_omnibus = add_fdr([secondary_result["omnibus"]])
        secondary_contrasts = add_fdr(secondary_result["contrasts"])
        rounded(secondary_omnibus).to_csv(output_dir / "tmtb_time_completers_omnibus_secondary.csv", index=False)
        rounded(secondary_contrasts).to_csv(output_dir / "tmtb_time_completers_contrasts_secondary.csv", index=False)
        continuous_for_plot["TMT-B time among completers"] = (secondary_result, tmtb_config)

    pd.DataFrame(failures, columns=["Outcome", "Analysis", "Reason"]).to_csv(
        output_dir / "cognitive_model_failures.csv", index=False
    )

    omnibus_q = primary_omnibus.set_index("Outcome")["q_FDR"] if not primary_omnibus.empty else pd.Series(dtype=float)
    for outcome, (result, config) in continuous_for_plot.items():
        if result["omnibus"]["Analysis"] == "Primary LMM":
            contrasts = primary_contrasts[primary_contrasts["Outcome"] == outcome]
            q_value = omnibus_q.loc[outcome]
            filename = f"cognitive_trajectory_{outcome.lower().replace(' ', '_').replace('-', '')}.png"
        else:
            contrasts = secondary_contrasts[secondary_contrasts["Outcome"] == outcome]
            q_value = secondary_omnibus.loc[0, "q_FDR"]
            filename = "tmtb_time_completers_trajectory_secondary.png"
        make_continuous_trajectory(outcome, config, result["long"], result["model_means"], q_value,
                                   contrasts, output_dir / filename)

    if completion_result is not None:
        completion_contrasts = primary_contrasts[
            primary_contrasts["Outcome"] == "TMT-B completion before 300 s"
        ]
        make_tmtb_completion_plot(
            completion_result, omnibus_q.loc["TMT-B completion before 300 s"], completion_contrasts,
            output_dir / "tmtb_completion_trajectory.png"
        )

    if not primary_omnibus.empty:
        make_cognitive_main_figure(
            continuous_for_plot, completion_result, primary_omnibus, primary_contrasts,
            secondary_omnibus, secondary_contrasts, output_dir / "cognitive_main_figure.png"
        )

    print(f"Saved results to: {output_dir}")
    print(f"Primary models fitted: {len(primary_results)}; model failures: {len(failures)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Longitudinal cognitive-test analysis for 0m, 1m and 3m.")
    parser.add_argument("csv_path", help="Wide cognitive-test CSV file")
    parser.add_argument("--output-dir", default="results_cognitive_lmm",
                        help="Output directory (default: results_cognitive_lmm)")
    arguments = parser.parse_args()
    try:
        main(arguments.csv_path, Path(arguments.output_dir))
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
