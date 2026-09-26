"""
Reviewer-responsive derivative-analysis pipeline for marble thermal conductivity.

Primary input
-------------
for derivatives paper.csv

Rock-ID grouping rule
---------------------
MA1–MA15 -> marble variety A
MB1–MB15 -> marble variety B
...
MZ1–MZ15 -> marble variety Z

Major analytical upgrades
-------------------------
1. Five-fold GroupKFold validation at marble-variety level.
2. Fixed-degree cluster bootstrap resampling entire marble varieties.
3. Exact zero curvature for degree-1 models.
4. Statistical consistency audit:
   Pearson r, standardized linear slope, r^2, linear R^2, and discrepancies.
5. Cubic-spline robustness benchmark under the same grouped folds.
6. Residual and influence diagnostics.
7. Sensitivity rankings both including and excluding thermal diffusivity.
8. Global-versus-within-variety normalization sensitivity analysis.
9. One-standard-error degree selection and degree-reselection bootstrap.
10. Validated within-variety normalization.
11. Leave-one-variety-out robustness and residual diagnostic figures.
12. Reviewer-comment matrix and reproducible Word/CSV outputs.

The contribution remains an integrated analytical workflow rather than a new
mathematical methodology.
"""

from __future__ import annotations

import argparse
import io
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from scipy import stats
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler


# ---------------------------------------------------------------------------
# USER SETTINGS
# ---------------------------------------------------------------------------
# These defaults are used when the script is run directly from Jupyter,
# Spyder, IPython, or a terminal without explicit command-line options.
DEFAULT_INPUT = "for derivatives paper.csv"
DEFAULT_OUTPUT_DIR = "upgraded_grouped_derivative_outputs"
DEFAULT_REPORT = "upgraded_grouped_derivative_report.docx"

TARGET = "thermal conductivity"
CANDIDATE_DEGREES = (1, 2, 3)
N_SPLITS = 5
N_BOOT = 3000
N_BOOT_RESELECT = 500
RANDOM_STATE = 42
GRID_POINTS = 200

plt.rcParams.update(
    {
        "font.size": 10,
        "font.weight": "bold",
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelweight": "bold",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linestyle": "--",
        "grid.linewidth": 0.55,
    }
)


@dataclass
class PolynomialFit:
    degree: int
    model: LinearRegression
    poly: PolynomialFeatures
    cv_mse_mean: float
    cv_mse_std: float
    cv_r2_mean: float
    cv_r2_std: float
    full_r2: float


def sanitize_name(name: str) -> str:
    name = str(name).strip().lower()
    replacements = {
        "thermal conductivity (w/m.k)": "thermal conductivity",
        "thermal conductivity (w/m·k)": "thermal conductivity",
        "thermal diffusivity (mm2/sec)": "thermal diffusivity",
        "thermal diffusivity (mm²/s)": "thermal diffusivity",
        "volumetric heat capacity (mj/m3k)": "volumetric heat capacity",
        "density (g/cc)": "density",
        "moisture content (%)": "moisture content",
        "water absorption (%)": "water absorption",
        "p-wave velocity (mps)": "p wave velocity",
        "p wave velocity (mps)": "p wave velocity",
        "ucs (mpa)": "ucs",
    }
    for old, new in replacements.items():
        name = name.replace(old, new)
    return re.sub(r"[^a-z0-9]+", " ", name).strip()


def load_and_validate(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [sanitize_name(c) for c in df.columns]

    if "rock id" not in df.columns:
        raise ValueError("Input must contain a 'Rock ID' column.")
    if TARGET not in df.columns:
        raise ValueError(f"Input must contain '{TARGET}'.")

    df["rock id"] = df["rock id"].astype(str).str.strip()
    df["marble variety"] = df["rock id"].map(extract_marble_variety)

    if df["marble variety"].isna().any():
        bad = df.loc[df["marble variety"].isna(), "rock id"].head(10).tolist()
        raise ValueError(
            "Could not derive marble variety from some Rock IDs. "
            f"Examples: {bad}. Expected IDs such as MA1, MB15, or MZ7."
        )

    for col in df.columns:
        if col not in {"rock id", "marble variety"}:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    numeric = [c for c in df.columns if c not in {"rock id", "marble variety"}]
    df = df.dropna(subset=[TARGET]).copy()

    if df["marble variety"].nunique() < N_SPLITS:
        raise ValueError(
            f"At least {N_SPLITS} marble varieties are required for {N_SPLITS}-fold GroupKFold."
        )

    # Report rather than silently discard property-specific missing values.
    if df[numeric].isna().all(axis=1).any():
        warnings.warn("Some rows contain no usable numeric predictors.")

    return df.reset_index(drop=True)


def extract_marble_variety(rock_id: str) -> str | None:
    """
    MA1 -> A, MB15 -> B, ..., MZ7 -> Z.
    Also accepts optional spaces, e.g., 'MA 1'.
    """
    match = re.fullmatch(r"\s*M([A-Z]+)\s*0*(\d+)\s*", str(rock_id).upper())
    return match.group(1) if match else None


def validate_group_structure(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("marble variety", as_index=False)
        .agg(
            n_samples=("rock id", "size"),
            first_id=("rock id", "first"),
            last_id=("rock id", "last"),
        )
        .sort_values("marble variety")
        .reset_index(drop=True)
    )
    summary["expected_15_samples"] = summary["n_samples"].eq(15)
    return summary


def group_folds(groups: np.ndarray, n_splits: int = N_SPLITS):
    splitter = GroupKFold(n_splits=n_splits)
    dummy = np.zeros(len(groups))
    return list(splitter.split(dummy, dummy, groups))


def grouped_polynomial_cv(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    degrees: Iterable[int] = CANDIDATE_DEGREES,
    n_splits: int = N_SPLITS,
) -> tuple[PolynomialFit, pd.DataFrame, np.ndarray]:
    """
    Select degree using the one-standard-error rule under GroupKFold.

    Entire marble varieties are withheld together. Predictions returned are
    out-of-fold predictions for the selected degree.
    """
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float).ravel()
    groups = np.asarray(groups)
    folds = group_folds(groups, n_splits)

    rows: list[dict] = []
    oof_by_degree: dict[int, np.ndarray] = {}

    for degree in degrees:
        oof = np.full(len(y), np.nan)
        fold_mse, fold_r2 = [], []

        for fold_no, (train_idx, test_idx) in enumerate(folds, start=1):
            poly = PolynomialFeatures(degree=degree, include_bias=False)
            x_train = poly.fit_transform(x[train_idx])
            x_test = poly.transform(x[test_idx])

            model = LinearRegression()
            model.fit(x_train, y[train_idx])
            pred = model.predict(x_test)
            oof[test_idx] = pred

            fold_mse.append(mean_squared_error(y[test_idx], pred))
            fold_r2.append(r2_score(y[test_idx], pred))

            rows.append(
                {
                    "degree": degree,
                    "fold": fold_no,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "n_train_groups": len(np.unique(groups[train_idx])),
                    "n_test_groups": len(np.unique(groups[test_idx])),
                    "fold_mse": fold_mse[-1],
                    "fold_r2": fold_r2[-1],
                    "test_groups": ", ".join(sorted(map(str, np.unique(groups[test_idx])))),
                }
            )

        oof_by_degree[degree] = oof

    fold_df = pd.DataFrame(rows)
    summary = (
        fold_df.groupby("degree", as_index=False)
        .agg(
            cv_mse_mean=("fold_mse", "mean"),
            cv_mse_std=("fold_mse", "std"),
            cv_r2_mean=("fold_r2", "mean"),
            cv_r2_std=("fold_r2", "std"),
            n_folds=("fold", "count"),
        )
    )
    summary["cv_mse_se"] = summary["cv_mse_std"] / np.sqrt(summary["n_folds"])
    min_row = summary.sort_values(["cv_mse_mean", "degree"]).iloc[0]
    threshold = float(min_row["cv_mse_mean"] + min_row["cv_mse_se"])
    summary["delta_from_min_mse"] = summary["cv_mse_mean"] - float(min_row["cv_mse_mean"])
    summary["one_se_eligible"] = summary["cv_mse_mean"] <= threshold
    best_degree = int(summary.loc[summary["one_se_eligible"], "degree"].min())
    best_row = summary.loc[summary["degree"] == best_degree].iloc[0]
    summary["selected_by_one_se"] = summary["degree"].eq(best_degree)

    poly = PolynomialFeatures(degree=best_degree, include_bias=False)
    xp = poly.fit_transform(x)
    model = LinearRegression().fit(xp, y)

    fit = PolynomialFit(
        degree=best_degree,
        model=model,
        poly=poly,
        cv_mse_mean=float(best_row["cv_mse_mean"]),
        cv_mse_std=float(best_row["cv_mse_std"]),
        cv_r2_mean=float(best_row["cv_r2_mean"]),
        cv_r2_std=float(best_row["cv_r2_std"]),
        full_r2=float(r2_score(y, model.predict(xp))),
    )
    return fit, fold_df, oof_by_degree[best_degree]


def polynomial_predict(fit: PolynomialFit, x_grid: np.ndarray) -> np.ndarray:
    xg = fit.poly.transform(np.asarray(x_grid).reshape(-1, 1))
    return fit.model.predict(xg)


def first_derivative(model: LinearRegression, x_grid: np.ndarray) -> np.ndarray:
    x = np.asarray(x_grid, dtype=float)
    out = np.zeros_like(x)
    for power, coefficient in enumerate(np.asarray(model.coef_).ravel(), start=1):
        out += power * coefficient * np.power(x, power - 1)
    return out


def second_derivative(model: LinearRegression, x_grid: np.ndarray) -> np.ndarray:
    x = np.asarray(x_grid, dtype=float)
    out = np.zeros_like(x)
    for power, coefficient in enumerate(np.asarray(model.coef_).ravel(), start=1):
        if power >= 2:
            out += power * (power - 1) * coefficient * np.power(x, power - 2)
    # Numerical guarantee for genuinely linear models.
    if len(np.asarray(model.coef_).ravel()) == 1:
        out[:] = 0.0
    return out


def cluster_bootstrap_derivatives(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    x_grid: np.ndarray,
    fixed_degree: int,
    n_boot: int = N_BOOT,
    random_state: int = RANDOM_STATE,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Resample whole marble varieties with replacement.

    The polynomial degree selected once by GroupKFold is held fixed in every
    replicate. This separates coefficient uncertainty from model-selection
    uncertainty and guarantees zero curvature for degree-1 models.
    """
    rng = np.random.default_rng(random_state)
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)

    d1_values = np.empty((n_boot, len(x_grid)), dtype=float)
    d2_values = np.empty((n_boot, len(x_grid)), dtype=float)

    for b in range(n_boot):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        sampled_indices = np.concatenate(
            [np.flatnonzero(groups == group) for group in sampled_groups]
        )

        xb = x[sampled_indices].reshape(-1, 1)
        yb = y[sampled_indices]

        poly = PolynomialFeatures(degree=fixed_degree, include_bias=False)
        xpb = poly.fit_transform(xb)
        model = LinearRegression().fit(xpb, yb)

        d1_values[b] = first_derivative(model, x_grid)
        d2_values[b] = second_derivative(model, x_grid)

    return summarize_bootstrap(d1_values), summarize_bootstrap(d2_values)


def summarize_bootstrap(values: np.ndarray) -> dict[str, np.ndarray]:
    positive = np.mean(values > 0, axis=0)
    negative = np.mean(values < 0, axis=0)
    zero = np.mean(np.isclose(values, 0.0, atol=1e-12), axis=0)
    return {
        "mean": np.mean(values, axis=0),
        "lo": np.percentile(values, 2.5, axis=0),
        "hi": np.percentile(values, 97.5, axis=0),
        "sign_stability": np.maximum.reduce([positive, negative, zero]),
    }


def consistency_audit(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """
    For standardized simple linear regression:
      slope ~= Pearson r
      R^2 ~= r^2
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    r = float(np.corrcoef(x, y)[0, 1])

    linear = LinearRegression().fit(x.reshape(-1, 1), y)
    slope = float(linear.coef_[0])
    linear_r2 = float(r2_score(y, linear.predict(x.reshape(-1, 1))))

    return {
        "pearson_r": r,
        "standardized_linear_slope": slope,
        "abs_r_minus_slope": abs(r - slope),
        "r_squared": r**2,
        "linear_full_r2": linear_r2,
        "abs_r2_minus_linear_r2": abs(r**2 - linear_r2),
    }


def residual_influence_diagnostics(
    x: np.ndarray,
    y: np.ndarray,
    fit: PolynomialFit,
    groups: np.ndarray,
) -> dict[str, float | str]:
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float).ravel()
    groups = np.asarray(groups)

    xp = fit.poly.transform(x)
    pred = fit.model.predict(xp)
    residuals = y - pred
    n = len(y)
    p = xp.shape[1] + 1

    design = np.column_stack([np.ones(n), xp])
    xtx_inv = np.linalg.pinv(design.T @ design)
    leverage = np.einsum("ij,jk,ik->i", design, xtx_inv, design)
    mse = np.sum(residuals**2) / max(n - p, 1)
    cooks = (residuals**2 / max(p * mse, np.finfo(float).eps)) * (
        leverage / np.maximum((1 - leverage) ** 2, np.finfo(float).eps)
    )

    worst_idx = int(np.nanargmax(cooks))
    abs_resid_fitted_corr = (
        float(np.corrcoef(np.abs(residuals), pred)[0, 1])
        if np.std(pred) > 0 and np.std(np.abs(residuals)) > 0
        else np.nan
    )

    return {
        "full_rmse": float(math.sqrt(mean_squared_error(y, pred))),
        "full_mae": float(mean_absolute_error(y, pred)),
        "residual_mean": float(np.mean(residuals)),
        "abs_residual_fitted_correlation": abs_resid_fitted_corr,
        "max_leverage": float(np.max(leverage)),
        "max_cooks_distance": float(np.max(cooks)),
        "most_influential_group": str(groups[worst_idx]),
        "most_influential_row_index": worst_idx,
    }


def grouped_spline_benchmark(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = N_SPLITS,
) -> dict[str, float]:
    """
    Flexible differentiable benchmark.

    Candidate cubic B-spline models are selected by grouped CV over the number
    of knots and ridge penalty. This is a robustness benchmark, not a
    replacement for the interpretable polynomial derivative model.
    """
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float).ravel()
    groups = np.asarray(groups)
    folds = group_folds(groups, n_splits)

    candidates = [(4, 0.01), (4, 0.1), (5, 0.01), (5, 0.1), (6, 0.1), (6, 1.0)]
    records = []

    for n_knots, alpha in candidates:
        fold_mse, fold_r2 = [], []
        oof = np.full(len(y), np.nan)

        for train_idx, test_idx in folds:
            model = make_pipeline(
                SplineTransformer(
                    n_knots=n_knots,
                    degree=3,
                    include_bias=False,
                    extrapolation="linear",
                ),
                StandardScaler(),
                Ridge(alpha=alpha),
            )
            model.fit(x[train_idx], y[train_idx])
            pred = model.predict(x[test_idx])
            oof[test_idx] = pred
            fold_mse.append(mean_squared_error(y[test_idx], pred))
            fold_r2.append(r2_score(y[test_idx], pred))

        records.append(
            {
                "spline_n_knots": n_knots,
                "spline_alpha": alpha,
                "spline_cv_mse_mean": float(np.mean(fold_mse)),
                "spline_cv_r2_mean": float(np.mean(fold_r2)),
                "spline_oof_r2": float(r2_score(y, oof)),
            }
        )

    return min(records, key=lambda item: (item["spline_cv_mse_mean"], item["spline_n_knots"]))


def within_variety_zscores(
    df: pd.DataFrame,
    numeric_columns: list[str],
) -> pd.DataFrame:
    out = df.copy()
    for col in numeric_columns:
        means = out.groupby("marble variety")[col].transform("mean")
        stds = out.groupby("marble variety")[col].transform("std")
        out[col] = (out[col] - means) / stds.replace(0, np.nan)
    return out


def grouped_normalization_sensitivity(
    global_df: pd.DataFrame,
    predictor: str,
) -> dict[str, float]:
    cols = [predictor, TARGET]
    raw = global_df[["rock id", "marble variety", *cols]].dropna().copy()
    within = raw.copy()
    for col in cols:
        within[f"{col}__within_z"] = within.groupby("marble variety")[col].transform(
            lambda v: (v-v.mean())/v.std(ddof=1) if v.std(ddof=1)>0 else np.nan
        )
    within = within.dropna(subset=[f"{predictor}__within_z", f"{TARGET}__within_z"])
    if len(within) == 0:
        return {"within_variety_best_degree":np.nan,"within_variety_cv_mse_mean":np.nan,"within_variety_cv_r2_mean":np.nan,"within_normalization_passed":False}
    checks=within.groupby("marble variety")[[f"{predictor}__within_z",f"{TARGET}__within_z"]].agg(['mean','std'])
    passed=bool(np.allclose(checks.xs('mean',axis=1,level=1),0,atol=1e-8) and np.allclose(checks.xs('std',axis=1,level=1),1,atol=1e-6))
    fit, _, _ = grouped_polynomial_cv(
        within[f"{predictor}__within_z"].to_numpy(),
        within[f"{TARGET}__within_z"].to_numpy(),
        within["marble variety"].to_numpy(),
    )
    return {"within_variety_best_degree":fit.degree,"within_variety_cv_mse_mean":fit.cv_mse_mean,"within_variety_cv_r2_mean":fit.cv_r2_mean,"within_normalization_passed":passed}



def normalization_diagnostics(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Validate true within-variety standardization for every requested column."""
    rows = []
    for col in columns:
        temp = df[["marble variety", col]].dropna().copy()
        z = temp.groupby("marble variety")[col].transform(
            lambda v: (v - v.mean()) / v.std(ddof=1) if v.std(ddof=1) > 0 else np.nan
        )
        temp["within_z"] = z
        chk = temp.groupby("marble variety")["within_z"].agg(["mean", "std"])
        rows.append({
            "variable": col,
            "n_groups_checked": int(len(chk)),
            "max_abs_group_mean": float(chk["mean"].abs().max()),
            "min_group_sd": float(chk["std"].min()),
            "max_group_sd": float(chk["std"].max()),
            "mean_group_sd": float(chk["std"].mean()),
            "normalization_check_passed": bool(
                np.allclose(chk["mean"].fillna(0), 0, atol=1e-8)
                and np.allclose(chk["std"].dropna(), 1, atol=1e-6)
            ),
        })
    return pd.DataFrame(rows)


def degree_reselection_cluster_bootstrap(
    x, y, groups, n_boot=N_BOOT_RESELECT, random_state=RANDOM_STATE
) -> dict[str, float]:
    """Quantify model-selection uncertainty by reselecting degree per cluster bootstrap."""
    rng = np.random.default_rng(random_state)
    x=np.asarray(x,float); y=np.asarray(y,float); groups=np.asarray(groups)
    unique=np.unique(groups); counts={d:0 for d in CANDIDATE_DEGREES}; failures=0
    for _ in range(n_boot):
        sampled=rng.choice(unique, size=len(unique), replace=True)
        idx=np.concatenate([np.flatnonzero(groups==g) for g in sampled])
        # create unique bootstrap cluster labels so duplicated varieties remain distinct clusters
        boot_groups=np.concatenate([
            np.repeat(f"{g}__{j}", np.sum(groups==g)) for j,g in enumerate(sampled)
        ])
        try:
            fit,_,_=grouped_polynomial_cv(x[idx], y[idx], boot_groups)
            counts[fit.degree]+=1
        except Exception:
            failures+=1
    valid=max(1,n_boot-failures)
    result={f"degree_{d}_selection_frequency": counts[d]/valid for d in CANDIDATE_DEGREES}
    result["reselection_bootstrap_valid"] = valid
    result["reselection_bootstrap_failures"] = failures
    return result


def leave_one_variety_out_analysis(x, y, groups) -> pd.DataFrame:
    """Assess whether degree, performance, and derivative direction depend on one variety."""
    rows=[]; x=np.asarray(x,float); y=np.asarray(y,float); groups=np.asarray(groups)
    for omitted in np.unique(groups):
        keep=groups!=omitted
        fit,_,oof=grouped_polynomial_cv(x[keep],y[keep],groups[keep])
        grid=np.linspace(np.min(x[keep]),np.max(x[keep]),GRID_POINTS)
        d1=first_derivative(fit.model,grid); d2=second_derivative(fit.model,grid)
        rows.append({
            "omitted_variety": omitted,
            "selected_degree": fit.degree,
            "grouped_cv_r2_mean": fit.cv_r2_mean,
            "grouped_oof_r2": float(r2_score(y[keep],oof)),
            "mean_first_derivative": float(np.mean(d1)),
            "mean_abs_first_derivative": float(np.mean(np.abs(d1))),
            "mean_second_derivative": float(np.mean(d2)),
            "first_derivative_sign": "positive" if np.mean(d1)>0 else "negative" if np.mean(d1)<0 else "zero",
        })
    return pd.DataFrame(rows)


def build_residual_diagnostic_figure(x, y, fit, groups, predictor):
    xp=fit.poly.transform(np.asarray(x).reshape(-1,1)); fitted=fit.model.predict(xp)
    resid=np.asarray(y)-fitted
    fig,axes=plt.subplots(2,2,figsize=(10,8),constrained_layout=True)
    axes[0,0].scatter(fitted,resid,s=18,alpha=.65); axes[0,0].axhline(0,ls='--')
    axes[0,0].set_title('Residuals vs fitted'); axes[0,0].set_xlabel('Fitted'); axes[0,0].set_ylabel('Residual')
    stats.probplot(resid, dist='norm', plot=axes[0,1]); axes[0,1].set_title('Normal Q-Q plot')
    axes[1,0].scatter(fitted,np.sqrt(np.abs(resid)),s=18,alpha=.65)
    axes[1,0].set_title('Scale-location'); axes[1,0].set_xlabel('Fitted'); axes[1,0].set_ylabel('sqrt(|residual|)')
    grp=pd.DataFrame({'group':groups,'resid':resid}).groupby('group')['resid'].mean()
    axes[1,1].bar(grp.index.astype(str),grp.values); axes[1,1].axhline(0,ls='--')
    axes[1,1].set_title('Mean residual by marble variety'); axes[1,1].tick_params(axis='x',rotation=90)
    fig.suptitle(f'Residual diagnostics: {predictor.title()}')
    return fig


def build_reviewer_matrix() -> pd.DataFrame:
    return pd.DataFrame([
        ["Specimen-level leakage", "GroupKFold by marble variety", "Methods/Validation", "grouped_cv_fold_scores.csv"],
        ["Dependent bootstrap", "Complete-variety cluster bootstrap", "Methods/Uncertainty", "primary_summary.csv"],
        ["Model complexity", "One-standard-error rule plus degree-reselection bootstrap", "Methods/Model selection", "degree_selection_summary.csv; model_selection_uncertainty.csv"],
        ["Linear curvature", "Exact zero second derivative and categorical validation flag", "Results/Derivatives", "primary_summary.csv"],
        ["Normalization", "Validated within-variety z-scores with assertions", "Sensitivity analysis", "normalization_validation.csv"],
        ["Influential varieties", "Leave-one-variety-out analysis", "Robustness", "leave_one_variety_out.csv"],
        ["Residual assumptions", "Residual, Q-Q, scale-location and group residual plots", "Diagnostics", "Word report figures"],
        ["Thermal diffusivity circularity", "Separate ranking and provenance warning", "Interpretive safeguards", "Word report"],
        ["Physical interpretation", "Manuscript action required; code cannot establish mechanism", "Discussion", "Reviewer response"],
        ["Experimental methods/uncertainty", "Manuscript action required", "Methods", "Reviewer response"],
    ],columns=["reviewer_issue","revision_or_action","manuscript_location","supporting_output"])

def figure_to_stream(fig) -> io.BytesIO:
    stream = io.BytesIO()
    fig.savefig(stream, format="png", dpi=350, bbox_inches="tight")
    stream.seek(0)
    plt.close(fig)
    return stream


PREDICTOR_FRAME_ORDER = [
    "density",
    "porosity",
    "moisture content",
    "water absorption",
    "rebound number",
    "p wave velocity",
    "ucs",
    "thermal diffusivity",
    "volumetric heat capacity",
]


def build_grouped_predictor_frame(
    plot_data: dict[str, dict],
    panel_type: str,
):
    """Build one 3 x 3 publication frame for response, first derivative, or second derivative."""
    if panel_type not in {"response", "first_derivative", "second_derivative"}:
        raise ValueError(f"Unsupported panel_type: {panel_type}")

    fig, axes = plt.subplots(3, 3, figsize=(15.5, 13.0), constrained_layout=True)
    axes = axes.ravel()

    for ax, predictor in zip(axes, PREDICTOR_FRAME_ORDER):
        if predictor not in plot_data:
            ax.set_visible(False)
            continue

        item = plot_data[predictor]
        x = item["x"]
        y = item["y"]
        x_grid = item["x_grid"]
        fit = item["fit"]

        if panel_type == "response":
            ax.scatter(x, y, s=13, alpha=0.55, label="Observed samples")
            ax.plot(
                x_grid,
                item["response"],
                linewidth=2.2,
                label=f"Polynomial degree {fit.degree}",
            )
            ax.set_ylabel("Thermal Conductivity (z-score)")
            ax.set_title(predictor.title())

        elif panel_type == "first_derivative":
            d1 = item["d1"]
            ax.fill_between(
                x_grid,
                d1["lo"],
                d1["hi"],
                alpha=0.20,
                label="95% cluster-bootstrap interval",
            )
            ax.plot(x_grid, d1["mean"], linewidth=2.2, label="Mean first derivative")
            ax.axhline(0, linestyle="--", linewidth=1)
            ax.set_ylabel("First derivative")
            ax.set_title(predictor.title())

        else:
            d2 = item["d2"]
            ax.fill_between(
                x_grid,
                d2["lo"],
                d2["hi"],
                alpha=0.20,
                label="95% cluster-bootstrap interval",
            )
            ax.plot(x_grid, d2["mean"], linewidth=2.2, label="Mean second derivative")
            ax.axhline(0, linestyle="--", linewidth=1)
            ax.set_ylabel("Second derivative")
            ax.set_title(predictor.title())

        ax.set_xlabel(f"{predictor.title()} (z-score)")
        ax.legend(loc="best", frameon=True)

    titles = {
        "response": "Thermal-Conductivity Response Profiles Across Nine Predictors",
        "first_derivative": "First-Order Derivative Sensitivity Profiles Across Nine Predictors",
        "second_derivative": "Second-Order Derivative Curvature Profiles Across Nine Predictors",
    }
    fig.suptitle(titles[panel_type], fontsize=17, fontweight="bold")
    return fig


def build_ranking_figure(summary: pd.DataFrame, exclude_diffusivity: bool):
    data = summary.copy()
    title_suffix = "excluding thermal diffusivity" if exclude_diffusivity else "all predictors"
    if exclude_diffusivity:
        data = data[data["predictor"] != "thermal diffusivity"]
    data = data.sort_values("mean_abs_first_derivative", ascending=True)

    fig, ax = plt.subplots(figsize=(8.6, 6.5), constrained_layout=True)
    ax.barh(data["predictor"].str.title(), data["mean_abs_first_derivative"])
    ax.set_xlabel("Mean absolute first derivative")
    ax.set_ylabel("Predictor")
    ax.set_title(f"Thermal-conductivity sensitivity ranking: {title_suffix}")
    return fig


def add_table(doc: Document, df: pd.DataFrame, title: str, max_rows: int | None = None):
    doc.add_heading(title, level=2)
    if df.empty:
        doc.add_paragraph("No results available.")
        return

    shown = df.head(max_rows).copy() if max_rows else df.copy()
    table = doc.add_table(rows=1, cols=len(shown.columns))
    table.style = "Table Grid"

    for i, col in enumerate(shown.columns):
        table.rows[0].cells[i].text = str(col)

    for _, row in shown.iterrows():
        cells = table.add_row().cells
        for i, value in enumerate(row):
            if isinstance(value, (float, np.floating)):
                cells[i].text = "" if pd.isna(value) else f"{float(value):.6f}"
            else:
                cells[i].text = str(value)

    if max_rows and len(df) > max_rows:
        doc.add_paragraph(f"Only the first {max_rows} rows are displayed.")


def add_figure(doc: Document, stream: io.BytesIO, caption: str):
    doc.add_picture(stream, width=Inches(6.3))
    paragraph = doc.add_paragraph(caption)
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER


def build_report(
    output_path: Path,
    group_summary: pd.DataFrame,
    summary: pd.DataFrame,
    cv_scores: pd.DataFrame,
    consistency: pd.DataFrame,
    diagnostics: pd.DataFrame,
    robustness: pd.DataFrame,
    normalization: pd.DataFrame,
    degree_selection: pd.DataFrame,
    model_selection_uncertainty: pd.DataFrame,
    normalization_validation: pd.DataFrame,
    lovo_summary: pd.DataFrame,
    reviewer_matrix: pd.DataFrame,
    figures: list[tuple[str, io.BytesIO]],
    n_boot: int,
):
    doc = Document()
    doc.add_heading("Grouped Derivative Analysis of Marble Thermal Conductivity", level=0)

    doc.add_paragraph(
        "This report implements a reviewer-responsive analytical workflow. "
        "Marble variety is derived from Rock ID (MA = variety A, MB = variety B, …, "
        "MZ = variety Z), and all samples belonging to the same variety remain in the "
        "same validation fold."
    )

    doc.add_heading("Implemented analytical revisions", level=1)
    revisions = [
        "Replaced shuffled specimen-level K-fold validation with five-fold GroupKFold validation by marble variety.",
        "Replaced row-wise bootstrap resampling with a cluster bootstrap that resamples complete marble varieties.",
        "Held the GroupKFold-selected polynomial degree fixed during bootstrap estimation.",
        "Guaranteed a mathematically exact zero second derivative for degree-1 models.",
        "Added Pearson-correlation, standardized-slope, and R² consistency checks.",
        "Added grouped cubic-spline benchmarking to test whether low-order polynomial conclusions are robust to a flexible model form.",
        "Added residual, leverage, and Cook's-distance diagnostics.",
        "Produced sensitivity rankings with and without thermal diffusivity.",
        "Added a validated global-versus-within-variety normalization sensitivity analysis.",
        "Applied the one-standard-error rule to favor parsimonious polynomial models.",
        "Added degree-reselection cluster bootstrap to quantify model-selection uncertainty.",
        "Added leave-one-variety-out robustness analysis and residual diagnostic plots.",
        f"Used {n_boot} fixed-degree cluster-bootstrap replicates for final derivative uncertainty estimation.",
    ]
    for item in revisions:
        doc.add_paragraph(item, style="List Bullet")

    doc.add_heading("Validation structure", level=1)
    doc.add_paragraph(
        "The validation target is generalization to previously unseen marble varieties, "
        "not merely to additional specimens from varieties already represented during training."
    )
    add_table(doc, group_summary, "Marble-variety structure")

    doc.add_heading("Integrated results", level=1)
    add_table(doc, summary, "Primary derivative and grouped-validation summary")
    add_table(doc, consistency, "Statistical consistency audit")
    add_table(doc, diagnostics, "Residual and influence diagnostics")
    add_table(doc, robustness, "Polynomial-versus-cubic-spline robustness comparison")
    add_table(doc, degree_selection, "Polynomial degree selection using the one-standard-error rule")
    add_table(doc, model_selection_uncertainty, "Model-selection uncertainty from degree-reselection cluster bootstrap")
    add_table(doc, normalization, "Normalization sensitivity analysis")
    add_table(doc, normalization_validation, "Within-variety normalization validation")
    add_table(doc, lovo_summary, "Leave-one-variety-out robustness summary")
    add_table(doc, reviewer_matrix, "Reviewer-comment coverage matrix")
    add_table(doc, cv_scores[cv_scores["selected_by_one_se"] == True] if "selected_by_one_se" in cv_scores.columns else cv_scores, "Selected-model fold-level GroupKFold results", max_rows=100)

    doc.add_heading("Interpretive safeguards", level=1)
    safeguards = [
        "Thermal diffusivity is retained as a thermophysical reference relationship but is also excluded from a separate ranking because it is physically related to thermal conductivity.",
        "Derivative inference is restricted to the observed predictor domain; no extrapolative physical claims are made.",
        "A strong derivative magnitude is not treated as equivalent to strong predictive performance; both sensitivity and grouped validation metrics must be considered.",
        "Spline benchmarking is used only as a robustness check and does not replace the polynomial model used for analytical derivatives.",
        "Fixed-degree bootstrap intervals are conditional on the one-SE-selected model; degree-reselection frequencies separately quantify model-selection uncertainty.",
        "Thermal diffusivity and volumetric heat capacity require explicit provenance in the manuscript; derived quantities must not be presented as fully independent predictors.",
    ]
    for item in safeguards:
        doc.add_paragraph(item, style="List Bullet")

    doc.add_heading("Figures", level=1)
    for caption, stream in figures:
        add_figure(doc, stream, caption)

    doc.add_heading("Reviewer-coverage statement", level=1)
    doc.add_paragraph(
        "These revisions directly address the principal analytical criticisms concerning "
        "specimen-level data leakage, bootstrap dependence, non-zero curvature reported for "
        "linear models, internal statistical consistency, imposed polynomial curvature, "
        "thermal-diffusivity circularity, normalization sensitivity, and influential observations. "
        "They do not by themselves resolve comments requiring expanded experimental documentation, "
        "additional literature synthesis, physical discussion, data/code archiving, or editorial rewriting."
    )

    doc.save(output_path)


def run_analysis(input_path: Path, output_dir: Path, report_name: str, n_boot: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_and_validate(input_path)
    group_summary = validate_group_structure(df)

    predictors = [
        c for c in df.columns
        if c not in {"rock id", "marble variety", TARGET}
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    summary_rows = []
    cv_frames = []
    consistency_rows = []
    diagnostic_rows = []
    robustness_rows = []
    normalization_rows = []
    degree_selection_frames = []
    model_selection_rows = []
    lovo_frames = []
    figures: list[tuple[str, io.BytesIO]] = []
    grouped_plot_data: dict[str, dict] = {}

    for predictor in predictors:
        sub = df[[predictor, TARGET, "marble variety"]].dropna().copy()
        x = sub[predictor].to_numpy()
        y = sub[TARGET].to_numpy()
        groups = sub["marble variety"].to_numpy()

        fit, fold_scores, oof = grouped_polynomial_cv(x, y, groups)
        fold_scores.insert(0, "predictor", predictor)
        # attach one-SE selection flag to every fold row
        fold_scores["selected_by_one_se"] = fold_scores["degree"].eq(fit.degree)
        cv_frames.append(fold_scores)
        degree_summary = fold_scores.groupby("degree",as_index=False).agg(
            cv_mse_mean=("fold_mse","mean"), cv_mse_std=("fold_mse","std"),
            cv_r2_mean=("fold_r2","mean"), cv_r2_std=("fold_r2","std"), n_folds=("fold","count"))
        degree_summary["cv_mse_se"] = degree_summary["cv_mse_std"]/np.sqrt(degree_summary["n_folds"])
        minrow=degree_summary.sort_values(["cv_mse_mean","degree"]).iloc[0]
        degree_summary["delta_from_min_mse"]=degree_summary["cv_mse_mean"]-minrow["cv_mse_mean"]
        degree_summary["one_se_eligible"]=degree_summary["cv_mse_mean"] <= minrow["cv_mse_mean"]+minrow["cv_mse_se"]
        degree_summary["selected_by_one_se"]=degree_summary["degree"].eq(fit.degree)
        degree_summary.insert(0,"predictor",predictor); degree_selection_frames.append(degree_summary)

        x_grid = np.linspace(np.min(x), np.max(x), GRID_POINTS)
        response = polynomial_predict(fit, x_grid)
        d1, d2 = cluster_bootstrap_derivatives(
            x=x,
            y=y,
            groups=groups,
            x_grid=x_grid,
            fixed_degree=fit.degree,
            n_boot=n_boot,
            random_state=RANDOM_STATE,
        )

        d1_mean_abs = float(np.mean(np.abs(d1["mean"])))
        d2_mean_abs = float(np.mean(np.abs(d2["mean"])))
        d1_stability = float(np.mean(d1["sign_stability"]))
        d2_stability = float(np.mean(d2["sign_stability"]))

        summary_rows.append(
            {
                "predictor": predictor,
                "n": len(sub),
                "n_groups": len(np.unique(groups)),
                "best_degree": fit.degree,
                "grouped_cv_mse_mean": fit.cv_mse_mean,
                "grouped_cv_mse_std": fit.cv_mse_std,
                "grouped_cv_r2_mean": fit.cv_r2_mean,
                "grouped_cv_r2_std": fit.cv_r2_std,
                "grouped_oof_r2": float(r2_score(y, oof)),
                "full_data_r2": fit.full_r2,
                "mean_abs_first_derivative": d1_mean_abs,
                "mean_abs_second_derivative": d2_mean_abs,
                "first_derivative_sign_stability": d1_stability,
                "second_derivative_sign_stability": d2_stability,
                "linear_zero_curvature_check": (
                    "Passed" if fit.degree == 1 and np.allclose(d2["mean"], 0.0, atol=1e-12)
                    else "Failed" if fit.degree == 1 else "Not applicable"
                ),
                "selected_model_has_nonzero_curvature": bool(not np.allclose(d2["mean"],0.0,atol=1e-12)),
            }
        )

        reselection = degree_reselection_cluster_bootstrap(x,y,groups,n_boot=min(N_BOOT_RESELECT,max(100,n_boot//3)),random_state=RANDOM_STATE)
        reselection["predictor"] = predictor
        model_selection_rows.append(reselection)

        lovo = leave_one_variety_out_analysis(x,y,groups)
        lovo.insert(0,"predictor",predictor)
        lovo_frames.append(lovo)

        audit = consistency_audit(x, y)
        audit["predictor"] = predictor
        consistency_rows.append(audit)

        diag = residual_influence_diagnostics(x, y, fit, groups)
        diag["predictor"] = predictor
        diagnostic_rows.append(diag)

        spline = grouped_spline_benchmark(x, y, groups)
        spline.update(
            {
                "predictor": predictor,
                "polynomial_degree": fit.degree,
                "polynomial_cv_mse_mean": fit.cv_mse_mean,
                "polynomial_cv_r2_mean": fit.cv_r2_mean,
                "spline_minus_polynomial_mse": (
                    spline["spline_cv_mse_mean"] - fit.cv_mse_mean
                ),
            }
        )
        robustness_rows.append(spline)

        norm = grouped_normalization_sensitivity(df, predictor)
        norm.update(
            {
                "predictor": predictor,
                "global_best_degree": fit.degree,
                "global_cv_mse_mean": fit.cv_mse_mean,
                "global_cv_r2_mean": fit.cv_r2_mean,
                "delta_within_minus_global_r2": (
                    norm["within_variety_cv_r2_mean"] - fit.cv_r2_mean
                ),
            }
        )
        normalization_rows.append(norm)

        grouped_plot_data[predictor] = {
            "x": x,
            "y": y,
            "x_grid": x_grid,
            "response": response,
            "d1": d1,
            "d2": d2,
            "fit": fit,
        }
        figures.append((f"Residual diagnostics for {predictor.title()}.", figure_to_stream(build_residual_diagnostic_figure(x,y,fit,groups,predictor))))

    summary = (
        pd.DataFrame(summary_rows)
        .sort_values("mean_abs_first_derivative", ascending=False)
        .reset_index(drop=True)
    )
    cv_scores = pd.concat(cv_frames, ignore_index=True)
    consistency = pd.DataFrame(consistency_rows)[
        [
            "predictor",
            "pearson_r",
            "standardized_linear_slope",
            "abs_r_minus_slope",
            "r_squared",
            "linear_full_r2",
            "abs_r2_minus_linear_r2",
        ]
    ]
    diagnostics = pd.DataFrame(diagnostic_rows)
    robustness = pd.DataFrame(robustness_rows)
    normalization = pd.DataFrame(normalization_rows)
    degree_selection = pd.concat(degree_selection_frames, ignore_index=True)
    model_selection_uncertainty = pd.DataFrame(model_selection_rows)
    lovo = pd.concat(lovo_frames, ignore_index=True)
    lovo_summary = lovo.groupby("predictor",as_index=False).agg(
        n_omissions=("omitted_variety","count"),
        degree_stability=("selected_degree",lambda v: float(np.mean(v==v.mode().iloc[0]))),
        min_grouped_cv_r2=("grouped_cv_r2_mean","min"),
        max_grouped_cv_r2=("grouped_cv_r2_mean","max"),
        derivative_sign_stability=("first_derivative_sign",lambda v: float(v.value_counts(normalize=True).max())),
    )
    normalization_validation = normalization_diagnostics(df, predictors+[TARGET])
    reviewer_matrix = build_reviewer_matrix()

    # Three consolidated 3 x 3 frames replace the nine former composite
    # response/first-derivative/second-derivative figures. Predictor order is:
    # row 1: density, porosity, moisture content;
    # row 2: water absorption, rebound number, p wave velocity;
    # row 3: ucs, thermal diffusivity, volumetric heat capacity.
    grouped_frames = [
        (
            "Response profiles for all nine predictors arranged in a 3 × 3 frame using the one-SE-selected polynomial models.",
            figure_to_stream(build_grouped_predictor_frame(grouped_plot_data, "response")),
        ),
        (
            "First-order derivative sensitivity profiles for all nine predictors arranged in a 3 × 3 frame with 95% variety-level cluster-bootstrap intervals.",
            figure_to_stream(build_grouped_predictor_frame(grouped_plot_data, "first_derivative")),
        ),
        (
            "Second-order derivative curvature profiles for all nine predictors arranged in a 3 × 3 frame with 95% variety-level cluster-bootstrap intervals.",
            figure_to_stream(build_grouped_predictor_frame(grouped_plot_data, "second_derivative")),
        ),
    ]
    figures = grouped_frames + figures

    # Two rankings: reference-inclusive and independent-proxy view.
    figures.insert(
        0,
        (
            "First-derivative sensitivity ranking across all predictors.",
            figure_to_stream(build_ranking_figure(summary, exclude_diffusivity=False)),
        ),
    )
    figures.insert(
        1,
        (
            "First-derivative sensitivity ranking excluding thermal diffusivity.",
            figure_to_stream(build_ranking_figure(summary, exclude_diffusivity=True)),
        ),
    )

    tables = {
        "group_structure.csv": group_summary,
        "primary_summary.csv": summary,
        "grouped_cv_fold_scores.csv": cv_scores,
        "statistical_consistency_audit.csv": consistency,
        "residual_influence_diagnostics.csv": diagnostics,
        "spline_robustness_comparison.csv": robustness,
        "normalization_sensitivity.csv": normalization,
        "normalization_validation.csv": normalization_validation,
        "degree_selection_summary.csv": degree_selection,
        "model_selection_uncertainty.csv": model_selection_uncertainty,
        "leave_one_variety_out.csv": lovo,
        "leave_one_variety_out_summary.csv": lovo_summary,
        "reviewer_comment_matrix.csv": reviewer_matrix,
    }
    for filename, table in tables.items():
        table.to_csv(output_dir / filename, index=False)

    report_path = output_dir / report_name
    build_report(
        output_path=report_path,
        group_summary=group_summary,
        summary=summary,
        cv_scores=cv_scores,
        consistency=consistency,
        diagnostics=diagnostics,
        robustness=robustness,
        normalization=normalization,
        degree_selection=degree_selection,
        model_selection_uncertainty=model_selection_uncertainty,
        normalization_validation=normalization_validation,
        lovo_summary=lovo_summary,
        reviewer_matrix=reviewer_matrix,
        figures=figures,
        n_boot=n_boot,
    )

    print(f"Analysis complete. Report: {report_path.resolve()}")
    return report_path


def parse_args(argv=None):
    """
    Parse supported options without failing inside Jupyter, Spyder, or IPython.

    Notebook kernels inject internal arguments such as:
        -f <path-to-kernel.json>

    parse_known_args() retains every supported pipeline argument and safely
    ignores only unrelated environment arguments. Terminal execution therefore
    remains fully supported.
    """
    parser = argparse.ArgumentParser(
        description="Grouped derivative analysis of marble thermal conductivity."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument(
        "--n-boot",
        type=int,
        default=N_BOOT,
        help="Number of marble-variety cluster-bootstrap replicates (default: 3000).",
    )

    args, unknown = parser.parse_known_args(args=argv)

    if unknown:
        kernel_only = (
            len(unknown) == 2
            and unknown[0] == "-f"
            and str(unknown[1]).lower().endswith(".json")
        )
        if not kernel_only:
            warnings.warn(
                "Ignoring unrecognized environment arguments: "
                + " ".join(map(str, unknown))
            )

    return args


def main():
    args = parse_args()
    if args.n_boot < 100:
        warnings.warn(
            "Fewer than 100 bootstrap replicates are suitable only for code testing, "
            "not for final manuscript results."
        )
    run_analysis(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        report_name=args.report,
        n_boot=args.n_boot,
    )


if __name__ == "__main__":
    main()
