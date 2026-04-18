from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import Any
import warnings as pywarnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.linalg import qr

from app_utils.dashboard_helpers import load_dashboard_bundle


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCENARIO_ORDER = ["Merit-based", "Moderate favoritism", "High nepotism risk"]

CANDIDATES_PATH = PROJECT_ROOT / "data" / "processed" / "candidates_model.csv"
EMPLOYEES_PATH = PROJECT_ROOT / "data" / "processed" / "employees_model.csv"

INFERENCE_CONFIG = {
    "hiring": {
        "module_path": "src.model_hiring",
        "target_column": "hired_flag",
        "output_path": PROJECT_ROOT / "outputs" / "hiring_model" / "hiring_explanatory_inference.csv",
        "title": "Hiring explanatory Logit",
    },
    "promotion": {
        "module_path": "src.model_promotion",
        "target_column": "promoted_flag",
        "output_path": PROJECT_ROOT / "outputs" / "promotion_model" / "promotion_explanatory_inference.csv",
        "title": "Promotion explanatory Logit",
    },
}


@dataclass
class OrganizationalImpactBundle:
    warnings: list[str]
    notes: list[str]
    proxy_column: str | None
    proxy_label: str
    proxy_metric_label: str
    executive_table: pd.DataFrame
    hiring_quality_table: pd.DataFrame
    promotion_fairness_table: pd.DataFrame
    efficiency_table: pd.DataFrame
    structural_risk_table: pd.DataFrame


def _sort_scenarios(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "scenario" not in df.columns:
        return df
    ordered = df.copy()
    ordered["scenario"] = pd.Categorical(ordered["scenario"], categories=SCENARIO_ORDER, ordered=True)
    ordered = ordered.sort_values("scenario").reset_index(drop=True)
    ordered["scenario"] = ordered["scenario"].astype(str)
    return ordered


def _safe_read_csv(path: Path) -> tuple[pd.DataFrame, str | None]:
    if not path.exists():
        return pd.DataFrame(), f"Missing data file: `{path.relative_to(PROJECT_ROOT)}`"
    try:
        return pd.read_csv(path), None
    except Exception as exc:
        return pd.DataFrame(), f"Could not read `{path.relative_to(PROJECT_ROOT)}`: {exc}"


def _safe_ratio(numerator: float | int | None, denominator: float | int | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    if pd.isna(numerator) or pd.isna(denominator) or float(denominator) == 0.0:
        return None
    return float(numerator) / float(denominator)


def _extract_anomaly_rates(summary_df: pd.DataFrame, raw_df: pd.DataFrame, flag_column: str) -> pd.DataFrame:
    if not summary_df.empty and {"scenario", "anomaly_rate"}.issubset(summary_df.columns):
        return _sort_scenarios(summary_df[["scenario", "anomaly_rate"]].copy())

    if not raw_df.empty and {"scenario", flag_column}.issubset(raw_df.columns):
        fallback = (
            raw_df.groupby("scenario", dropna=False)[flag_column]
            .mean()
            .reset_index(name="anomaly_rate")
        )
        return _sort_scenarios(fallback)

    return pd.DataFrame(columns=["scenario", "anomaly_rate"])


def _choose_proxy_column(employees_df: pd.DataFrame) -> tuple[str | None, str, str]:
    if "production" in employees_df.columns:
        return "production", "Production", "Production per 100 employees"
    if "performance_score" in employees_df.columns:
        return "performance_score", "Performance score proxy", "Performance-score proxy per 100 employees"
    if "merit_score" in employees_df.columns:
        return "merit_score", "Merit/performance proxy", "Merit/performance proxy per 100 employees"
    return None, "No production-style field found", "No production-style field found"


def _compute_hiring_quality(
    candidates_df: pd.DataFrame,
    employees_df: pd.DataFrame,
    hiring_anomaly_rates: pd.DataFrame,
) -> pd.DataFrame:
    required_candidate_columns = {"scenario", "merit_score"}
    required_employee_columns = {"scenario", "merit_score"}
    if not required_candidate_columns.issubset(candidates_df.columns) or not required_employee_columns.issubset(employees_df.columns):
        return pd.DataFrame()

    pool = (
        candidates_df.groupby("scenario", dropna=False)["merit_score"]
        .agg(candidate_pool_avg_merit="mean", candidate_pool_median_merit="median")
        .reset_index()
    )

    hired_rows = []
    for scenario_name, scenario_df in employees_df.groupby("scenario", dropna=False):
        pool_row = pool[pool["scenario"] == scenario_name]
        if pool_row.empty:
            continue
        candidate_pool_avg = float(pool_row["candidate_pool_avg_merit"].iloc[0])
        candidate_pool_median = float(pool_row["candidate_pool_median_merit"].iloc[0])
        hired_merit = pd.to_numeric(scenario_df["merit_score"], errors="coerce").dropna()
        hired_rows.append(
            {
                "scenario": scenario_name,
                "avg_hired_merit": float(hired_merit.mean()) if not hired_merit.empty else np.nan,
                "share_hires_below_candidate_pool_median": float((hired_merit < candidate_pool_median).mean())
                if not hired_merit.empty
                else np.nan,
                "candidate_pool_avg_merit": candidate_pool_avg,
                "hiring_quality_ratio": _safe_ratio(hired_merit.mean(), candidate_pool_avg),
            }
        )

    quality = pd.DataFrame(hired_rows)
    if quality.empty:
        return quality

    if not hiring_anomaly_rates.empty:
        quality = quality.merge(hiring_anomaly_rates, on="scenario", how="left")
        quality = quality.rename(columns={"anomaly_rate": "hiring_anomaly_rate"})

    return _sort_scenarios(quality)


def _compute_promotion_fairness(employees_df: pd.DataFrame, proxy_column: str | None) -> pd.DataFrame:
    if proxy_column is None:
        return pd.DataFrame()
    required_columns = {"scenario", "promoted_flag", proxy_column}
    if not required_columns.issubset(employees_df.columns):
        return pd.DataFrame()

    rows = []
    for scenario_name, scenario_df in employees_df.groupby("scenario", dropna=False):
        promoted = scenario_df[scenario_df["promoted_flag"] == 1]
        non_promoted = scenario_df[scenario_df["promoted_flag"] == 0]
        promoted_proxy = pd.to_numeric(promoted[proxy_column], errors="coerce").dropna()
        non_promoted_proxy = pd.to_numeric(non_promoted[proxy_column], errors="coerce").dropna()
        all_proxy = pd.to_numeric(scenario_df[proxy_column], errors="coerce").dropna()

        non_promoted_median = float(non_promoted_proxy.median()) if not non_promoted_proxy.empty else np.nan
        top_quartile_threshold = float(all_proxy.quantile(0.75)) if not all_proxy.empty else np.nan
        stronger_non_promoted_max = float(non_promoted_proxy.max()) if not non_promoted_proxy.empty else np.nan

        top_quartile_mask = pd.to_numeric(scenario_df[proxy_column], errors="coerce") >= top_quartile_threshold
        top_quartile_total = int(top_quartile_mask.sum())
        top_quartile_not_promoted = int((top_quartile_mask & (scenario_df["promoted_flag"] == 0)).sum())

        rows.append(
            {
                "scenario": scenario_name,
                "promotion_quality_ratio": _safe_ratio(promoted_proxy.mean(), non_promoted_proxy.mean()),
                "share_promoted_below_non_promoted_median": float((promoted_proxy < non_promoted_median).mean())
                if not promoted_proxy.empty and pd.notna(non_promoted_median)
                else np.nan,
                "promoted_outperformed_by_non_promoted_count": int((promoted_proxy < stronger_non_promoted_max).sum())
                if not promoted_proxy.empty and pd.notna(stronger_non_promoted_max)
                else 0,
                "promoted_outperformed_by_non_promoted_rate": float((promoted_proxy < stronger_non_promoted_max).mean())
                if not promoted_proxy.empty and pd.notna(stronger_non_promoted_max)
                else np.nan,
                "top_performer_neglect_rate": _safe_ratio(top_quartile_not_promoted, top_quartile_total),
            }
        )

    return _sort_scenarios(pd.DataFrame(rows))


def _compute_efficiency(employees_df: pd.DataFrame, proxy_column: str | None) -> pd.DataFrame:
    if proxy_column is None:
        return pd.DataFrame()
    required_columns = {"scenario", proxy_column}
    if not required_columns.issubset(employees_df.columns):
        return pd.DataFrame()

    rows = []
    for scenario_name, scenario_df in employees_df.groupby("scenario", dropna=False):
        proxy_values = pd.to_numeric(scenario_df[proxy_column], errors="coerce").dropna()
        salary_values = pd.to_numeric(scenario_df["salary"], errors="coerce").dropna() if "salary" in scenario_df.columns else pd.Series(dtype=float)
        total_proxy = float(proxy_values.sum()) if not proxy_values.empty else np.nan
        employee_count = int(proxy_values.shape[0])
        rows.append(
            {
                "scenario": scenario_name,
                "employee_count": employee_count,
                "total_proxy_output": total_proxy,
                "proxy_per_employee": float(proxy_values.mean()) if employee_count else np.nan,
                "proxy_per_100_employees": float(proxy_values.mean() * 100.0) if employee_count else np.nan,
                "proxy_per_salary_dollar": _safe_ratio(total_proxy, float(salary_values.sum()) if not salary_values.empty else None),
                "proxy_per_100_salary_dollars": (
                    _safe_ratio(total_proxy, float(salary_values.sum()) if not salary_values.empty else None) * 100.0
                    if not salary_values.empty
                    else None
                ),
                "proxy_per_10000_salary_dollars": (
                    _safe_ratio(total_proxy, float(salary_values.sum()) if not salary_values.empty else None) * 10000.0
                    if not salary_values.empty
                    else None
                ),
            }
        )

    return _sort_scenarios(pd.DataFrame(rows))


def _compute_structural_risk() -> tuple[pd.DataFrame, list[str]]:
    dashboard_bundle = load_dashboard_bundle()
    notes = list(dashboard_bundle.notes)

    network = dashboard_bundle.network_summary.copy()
    hiring_summary = dashboard_bundle.hiring_summary.copy()
    promotion_summary = dashboard_bundle.promotion_summary.copy()

    structural = pd.DataFrame({"scenario": SCENARIO_ORDER})
    if not network.empty:
        structural = structural.merge(network, on="scenario", how="left")
    if not hiring_summary.empty and {"scenario", "anomaly_rate"}.issubset(hiring_summary.columns):
        structural = structural.merge(
            hiring_summary[["scenario", "anomaly_rate"]].rename(columns={"anomaly_rate": "hiring_anomaly_rate"}),
            on="scenario",
            how="left",
        )
    if not promotion_summary.empty and {"scenario", "anomaly_rate"}.issubset(promotion_summary.columns):
        structural = structural.merge(
            promotion_summary[["scenario", "anomaly_rate"]].rename(columns={"anomaly_rate": "promotion_anomaly_rate"}),
            on="scenario",
            how="left",
        )

    return _sort_scenarios(structural), notes


@lru_cache(maxsize=1)
def _load_organizational_impact_bundle_cached() -> OrganizationalImpactBundle:
    warnings: list[str] = []
    notes: list[str] = []

    candidates_df, candidates_warning = _safe_read_csv(CANDIDATES_PATH)
    employees_df, employees_warning = _safe_read_csv(EMPLOYEES_PATH)
    if candidates_warning:
        warnings.append(candidates_warning)
    if employees_warning:
        warnings.append(employees_warning)

    if candidates_df.empty or employees_df.empty:
        return OrganizationalImpactBundle(
            warnings=warnings,
            notes=notes,
            proxy_column=None,
            proxy_label="No production-style field found",
            proxy_metric_label="No production-style field found",
            executive_table=pd.DataFrame(),
            hiring_quality_table=pd.DataFrame(),
            promotion_fairness_table=pd.DataFrame(),
            efficiency_table=pd.DataFrame(),
            structural_risk_table=pd.DataFrame(),
        )

    dashboard_bundle = load_dashboard_bundle()
    proxy_column, proxy_label, proxy_metric_label = _choose_proxy_column(employees_df)
    if proxy_column != "production":
        notes.append(
            f"`{proxy_label}` is used as the organizational-output proxy because no explicit `production` field is present in the current employee dataset."
        )

    hiring_rates = _extract_anomaly_rates(
        dashboard_bundle.hiring_summary,
        dashboard_bundle.frames.get("hiring_scores", pd.DataFrame()),
        "hiring_anomaly_flag",
    )
    promotion_rates = _extract_anomaly_rates(
        dashboard_bundle.promotion_summary,
        dashboard_bundle.frames.get("promotion_scores", pd.DataFrame()),
        "promotion_anomaly_flag",
    )

    hiring_quality = _compute_hiring_quality(candidates_df, employees_df, hiring_rates)
    promotion_fairness = _compute_promotion_fairness(employees_df, proxy_column)
    efficiency = _compute_efficiency(employees_df, proxy_column)
    structural_risk, structural_notes = _compute_structural_risk()
    notes.extend(structural_notes)

    executive = pd.DataFrame({"scenario": SCENARIO_ORDER})
    if not efficiency.empty and {"scenario", "proxy_per_100_employees"}.issubset(efficiency.columns):
        executive = executive.merge(efficiency[["scenario", "proxy_per_100_employees"]], on="scenario", how="left")
    if not hiring_rates.empty:
        executive = executive.merge(
            hiring_rates.rename(columns={"anomaly_rate": "hiring_anomaly_rate"}),
            on="scenario",
            how="left",
        )
    if not promotion_rates.empty:
        executive = executive.merge(
            promotion_rates.rename(columns={"anomaly_rate": "promotion_anomaly_rate"}),
            on="scenario",
            how="left",
        )

    return OrganizationalImpactBundle(
        warnings=warnings + list(dashboard_bundle.warnings),
        notes=notes,
        proxy_column=proxy_column,
        proxy_label=proxy_label,
        proxy_metric_label=proxy_metric_label,
        executive_table=_sort_scenarios(executive),
        hiring_quality_table=hiring_quality,
        promotion_fairness_table=promotion_fairness,
        efficiency_table=efficiency,
        structural_risk_table=structural_risk,
    )


def load_organizational_impact_bundle() -> OrganizationalImpactBundle:
    cached = _load_organizational_impact_bundle_cached()
    return OrganizationalImpactBundle(
        warnings=list(cached.warnings),
        notes=list(cached.notes),
        proxy_column=cached.proxy_column,
        proxy_label=cached.proxy_label,
        proxy_metric_label=cached.proxy_metric_label,
        executive_table=cached.executive_table.copy(),
        hiring_quality_table=cached.hiring_quality_table.copy(),
        promotion_fairness_table=cached.promotion_fairness_table.copy(),
        efficiency_table=cached.efficiency_table.copy(),
        structural_risk_table=cached.structural_risk_table.copy(),
    )


def _get_module(task: str):
    return import_module(INFERENCE_CONFIG[task]["module_path"])


def _current_inference_signature(task: str) -> tuple[int, int]:
    module = _get_module(task)
    module_path = Path(module.__file__).resolve()
    input_path = module.get_paths()["input"]
    return int(module_path.stat().st_mtime_ns), int(input_path.stat().st_mtime_ns)


def _reduce_collinearity(feature_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    reduced = feature_df.copy()
    dropped: list[str] = []

    zero_variance = [column for column in reduced.columns if reduced[column].nunique(dropna=False) <= 1]
    if zero_variance:
        reduced = reduced.drop(columns=zero_variance)
        dropped.extend(zero_variance)

    if reduced.empty:
        return reduced, dropped

    matrix = reduced.to_numpy(dtype=float)
    _, r_matrix, pivots = qr(matrix, mode="economic", pivoting=True)
    diagonal = np.abs(np.diag(r_matrix))
    tolerance = max(matrix.shape) * np.finfo(float).eps * (diagonal.max() if diagonal.size else 1.0)
    rank = int((diagonal > tolerance).sum())
    keep_indices = sorted(int(index) for index in pivots[:rank])
    keep_columns = [reduced.columns[index] for index in keep_indices]
    dropped.extend([column for column in reduced.columns if column not in keep_columns])
    return reduced[keep_columns].copy(), dropped


def _fit_explanatory_inference(task: str) -> tuple[pd.DataFrame, list[str]]:
    module = _get_module(task)
    config = INFERENCE_CONFIG[task]
    warnings: list[str] = []

    raw_df = pd.read_csv(module.get_paths()["input"])
    model_df = module.ensure_model_features(raw_df)
    target_column = config["target_column"]
    feature_schema = module.fit_feature_schema(model_df, list(module.EXPLANATORY_FEATURES), [])
    feature_df = module.transform_features(model_df, feature_schema)
    feature_df = feature_df.apply(pd.to_numeric, errors="coerce").fillna(feature_schema.medians)
    feature_df, dropped_columns = _reduce_collinearity(feature_df)

    if dropped_columns:
        warnings.append(
            "Dropped perfectly collinear or zero-variance explanatory columns for stable statsmodels inference: "
            + ", ".join(sorted(dropped_columns))
        )

    if feature_df.empty:
        raise ValueError("No explanatory columns remained after collinearity checks.")

    X = sm.add_constant(feature_df.astype(float), has_constant="add")
    y = pd.to_numeric(model_df[target_column], errors="raise").astype(int)
    model = sm.Logit(y, X)

    fit_errors: list[str] = []
    result = None
    for method in ("lbfgs", "bfgs", "newton"):
        try:
            with pywarnings.catch_warnings(record=True) as caught_warnings:
                pywarnings.simplefilter("always")
                candidate_result = model.fit(method=method, disp=False, maxiter=300)
            converged = bool(getattr(candidate_result, "mle_retvals", {}).get("converged", True))
            if converged:
                result = candidate_result
                for caught in caught_warnings:
                    message = str(caught.message)
                    if "failed to converge" in message.lower():
                        warnings.append(f"Statsmodels reported a convergence warning under `{method}`: {message}")
                break
            fit_errors.append(f"{method}: optimizer did not converge")
        except Exception as exc:  # pragma: no cover - runtime fallback
            fit_errors.append(f"{method}: {exc}")

    if result is None:
        raise RuntimeError("Statsmodels Logit fit failed. " + " | ".join(fit_errors))

    conf_int = result.conf_int()
    table = pd.DataFrame(
        {
            "variable": result.params.index.astype(str),
            "coefficient": result.params.values,
            "standard_error": result.bse.values,
            "z_stat": result.tvalues.values,
            "p_value": result.pvalues.values,
            "confidence_interval_lower": conf_int.iloc[:, 0].values,
            "confidence_interval_upper": conf_int.iloc[:, 1].values,
        }
    )
    table["significance_5pct"] = table["p_value"] < 0.05
    table["model"] = config["title"]
    table["task"] = task
    table["n_obs"] = int(result.nobs)
    table["pseudo_r_squared"] = float(result.prsquared) if hasattr(result, "prsquared") else np.nan
    return table, warnings


@lru_cache(maxsize=8)
def _load_explanatory_inference_cached(task: str, module_mtime_ns: int, input_mtime_ns: int) -> dict[str, Any]:
    del module_mtime_ns, input_mtime_ns
    config = INFERENCE_CONFIG[task]
    output_path = config["output_path"]

    try:
        frame, warnings = _fit_explanatory_inference(task)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_path, index=False)
        return {
            "frame": frame,
            "warnings": warnings,
            "note": f"Saved current statsmodels inference output to `{output_path.relative_to(PROJECT_ROOT)}`.",
            "output_path": str(output_path),
        }
    except Exception as exc:
        if output_path.exists():
            frame = pd.read_csv(output_path)
            return {
                "frame": frame,
                "warnings": [f"Current statsmodels inference refit failed: {exc}. Showing the last saved output instead."],
                "note": f"Loaded last saved statsmodels inference output from `{output_path.relative_to(PROJECT_ROOT)}`.",
                "output_path": str(output_path),
            }
        return {
            "frame": pd.DataFrame(),
            "warnings": [f"Statsmodels inference could not be generated for {task}: {exc}"],
            "note": "No saved inference table is available.",
            "output_path": str(output_path),
        }


def load_explanatory_inference(task: str, module_mtime_ns: int, input_mtime_ns: int) -> dict[str, Any]:
    cached = _load_explanatory_inference_cached(task, module_mtime_ns, input_mtime_ns)
    return {
        "frame": cached["frame"].copy(),
        "warnings": list(cached["warnings"]),
        "note": cached["note"],
        "output_path": cached["output_path"],
    }


def get_explanatory_inference(task: str) -> dict[str, Any]:
    module_mtime_ns, input_mtime_ns = _current_inference_signature(task)
    return load_explanatory_inference(task, module_mtime_ns, input_mtime_ns)


def format_coefficient_table(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    desired_columns = [
        "variable",
        "coefficient",
        "standard_error",
        "z_stat",
        "p_value",
        "confidence_interval_lower",
        "confidence_interval_upper",
        "significance_5pct",
    ]
    available = [column for column in desired_columns if column in frame.columns]
    formatted = frame[available].copy()
    rename_map = {
        "variable": "Variable",
        "coefficient": "Coefficient",
        "standard_error": "Std. error",
        "z_stat": "z-stat",
        "p_value": "p-value",
        "confidence_interval_lower": "CI lower",
        "confidence_interval_upper": "CI upper",
        "significance_5pct": "p < 0.05",
    }
    formatted = formatted.rename(columns=rename_map)
    numeric_columns = ["Coefficient", "Std. error", "z-stat", "p-value", "CI lower", "CI upper"]
    for column in numeric_columns:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(lambda value: f"{value:,.4f}" if pd.notna(value) else "N/A")
    if "p < 0.05" in formatted.columns:
        formatted["p < 0.05"] = formatted["p < 0.05"].map(lambda value: "Yes" if bool(value) else "")
    return formatted
