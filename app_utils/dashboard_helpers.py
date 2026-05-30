from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_PATHS = {
    "manager_scores": PROJECT_ROOT / "outputs" / "network_model" / "manager_nepotism_scores.csv",
    "department_scores": PROJECT_ROOT / "outputs" / "network_model" / "department_nepotism_scores.csv",
    "top_managers": PROJECT_ROOT / "outputs" / "network_model" / "top_risky_managers.csv",
    "top_departments": PROJECT_ROOT / "outputs" / "network_model" / "top_risky_departments.csv",
    "manager_chart": PROJECT_ROOT / "outputs" / "network_model" / "top_risky_managers.png",
    "department_chart": PROJECT_ROOT / "outputs" / "network_model" / "top_risky_departments.png",
    "hiring_scores": PROJECT_ROOT / "outputs" / "anomaly_model" / "hiring_anomaly_scores.csv",
    "promotion_scores": PROJECT_ROOT / "outputs" / "anomaly_model" / "promotion_anomaly_scores.csv",
    "top_hiring": PROJECT_ROOT / "outputs" / "anomaly_model" / "top_hiring_anomalies.csv",
    "top_promotion": PROJECT_ROOT / "outputs" / "anomaly_model" / "top_promotion_anomalies.csv",
    "hiring_summary": PROJECT_ROOT / "outputs" / "anomaly_model" / "hiring_anomaly_rate_by_scenario.csv",
    "promotion_summary": PROJECT_ROOT / "outputs" / "anomaly_model" / "promotion_anomaly_rate_by_scenario.csv",
    "hiring_chart": PROJECT_ROOT / "outputs" / "anomaly_model" / "hiring_anomaly_scores.png",
    "promotion_chart": PROJECT_ROOT / "outputs" / "anomaly_model" / "promotion_anomaly_scores.png",
}


@dataclass
class DashboardBundle:
    frames: dict[str, pd.DataFrame]
    image_paths: dict[str, Path]
    warnings: list[str]
    notes: list[str]
    network_metrics: list[dict[str, str]]
    network_summary: pd.DataFrame
    anomaly_metrics: list[dict[str, str]]
    hiring_summary: pd.DataFrame
    promotion_summary: pd.DataFrame


def _safe_read_csv(path: Path, warnings: list[str]) -> pd.DataFrame | None:
    if not path.exists():
        warnings.append(f"Missing output file: `{path.relative_to(PROJECT_ROOT)}`")
        return None

    try:
        return pd.read_csv(path)
    except Exception as exc:
        warnings.append(f"Could not read `{path.relative_to(PROJECT_ROOT)}`: {exc}")
        return None


@lru_cache(maxsize=1)
def load_dashboard_bundle_core() -> DashboardBundle:
    warnings: list[str] = []
    notes: list[str] = []

    frames: dict[str, pd.DataFrame] = {}
    required_keys = [
        "manager_scores",
        "department_scores",
        "top_managers",
        "top_departments",
        "hiring_scores",
        "promotion_scores",
        "top_hiring",
        "top_promotion",
        "hiring_summary",
        "promotion_summary",
    ]
    for key in required_keys:
        frame = _safe_read_csv(OUTPUT_PATHS[key], warnings)
        if frame is not None:
            frames[key] = frame

    image_paths = {
        key: path
        for key, path in OUTPUT_PATHS.items()
        if path.suffix.lower() == ".png" and path.exists()
    }

    return DashboardBundle(
        frames=frames,
        image_paths=image_paths,
        warnings=warnings,
        notes=notes,
        network_metrics=_compute_network_metrics(frames),
        network_summary=_compute_network_scenario_averages(frames),
        anomaly_metrics=_compute_anomaly_metrics(frames),
        hiring_summary=_sort_scenarios(frames.get("hiring_summary", pd.DataFrame()).copy())
        if "hiring_summary" in frames
        else pd.DataFrame(),
        promotion_summary=_sort_scenarios(frames.get("promotion_summary", pd.DataFrame()).copy())
        if "promotion_summary" in frames
        else pd.DataFrame(),
    )


def load_dashboard_bundle() -> DashboardBundle:
    return load_dashboard_bundle_core()


def _compute_network_metrics(frames: dict[str, pd.DataFrame]) -> list[dict[str, str]]:
    metrics: list[dict[str, str]] = []
    manager_scores = frames.get("manager_scores")
    department_scores = frames.get("department_scores")

    if manager_scores is not None and "manager_nepotism_risk_score" in manager_scores.columns:
        metrics.append(
            {
                "label": "Avg manager risk",
                "value": f"{manager_scores['manager_nepotism_risk_score'].mean():.3f}",
            }
        )
        top_row = manager_scores.sort_values("manager_nepotism_risk_score", ascending=False).iloc[0]
        metrics.append(
            {
                "label": "Top manager risk",
                "value": f"{top_row['manager_nepotism_risk_score']:.3f}",
                "help": f"{top_row.get('manager_full_name', top_row.get('manager_id', 'Manager'))}",
            }
        )

    if department_scores is not None and "department_nepotism_risk_score" in department_scores.columns:
        metrics.append(
            {
                "label": "Avg department risk",
                "value": f"{department_scores['department_nepotism_risk_score'].mean():.3f}",
            }
        )
        top_row = department_scores.sort_values("department_nepotism_risk_score", ascending=False).iloc[0]
        metrics.append(
            {
                "label": "Top department risk",
                "value": f"{top_row['department_nepotism_risk_score']:.3f}",
                "help": f"{top_row.get('company_id', '')} / {top_row.get('department_id', '')}",
            }
        )

    return metrics


def compute_network_metrics(bundle: DashboardBundle) -> list[dict[str, str]]:
    return list(bundle.network_metrics)


def _compute_network_scenario_averages(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    manager_scores = frames.get("manager_scores")
    department_scores = frames.get("department_scores")
    output: pd.DataFrame | None = None

    if manager_scores is not None and {"scenario", "manager_nepotism_risk_score"}.issubset(manager_scores.columns):
        manager_summary = (
            manager_scores.groupby("scenario", dropna=False)["manager_nepotism_risk_score"]
            .mean()
            .reset_index(name="avg_manager_risk_score")
        )
        output = manager_summary

    if department_scores is not None and {"scenario", "department_nepotism_risk_score"}.issubset(department_scores.columns):
        department_summary = (
            department_scores.groupby("scenario", dropna=False)["department_nepotism_risk_score"]
            .mean()
            .reset_index(name="avg_department_risk_score")
        )
        output = department_summary if output is None else output.merge(department_summary, on="scenario", how="outer")

    if output is None:
        return pd.DataFrame()
    return _sort_scenarios(output)


def compute_network_scenario_averages(bundle: DashboardBundle) -> pd.DataFrame:
    return bundle.network_summary.copy()


def _compute_anomaly_metrics(frames: dict[str, pd.DataFrame]) -> list[dict[str, str]]:
    metrics: list[dict[str, str]] = []
    hiring_scores = frames.get("hiring_scores")
    promotion_scores = frames.get("promotion_scores")

    if hiring_scores is not None:
        if "hiring_threshold" in hiring_scores.columns:
            metrics.append(
                {
                    "label": "Hiring threshold",
                    "value": f"{pd.to_numeric(hiring_scores['hiring_threshold'], errors='coerce').dropna().iloc[0]:.3f}",
                }
            )
        if "hiring_anomaly_flag" in hiring_scores.columns:
            metrics.append(
                {
                    "label": "Suspicious hires",
                    "value": f"{int(pd.to_numeric(hiring_scores['hiring_anomaly_flag'], errors='coerce').fillna(0).sum()):,}",
                }
            )

    if promotion_scores is not None:
        if "promotion_threshold" in promotion_scores.columns:
            metrics.append(
                {
                    "label": "Promotion threshold",
                    "value": f"{pd.to_numeric(promotion_scores['promotion_threshold'], errors='coerce').dropna().iloc[0]:.3f}",
                }
            )
        if "promotion_anomaly_flag" in promotion_scores.columns:
            metrics.append(
                {
                    "label": "Suspicious promotions",
                    "value": f"{int(pd.to_numeric(promotion_scores['promotion_anomaly_flag'], errors='coerce').fillna(0).sum()):,}",
                }
            )

    return metrics


def compute_anomaly_metrics(bundle: DashboardBundle) -> list[dict[str, str]]:
    return list(bundle.anomaly_metrics)


def compute_anomaly_summaries(bundle: DashboardBundle) -> tuple[pd.DataFrame, pd.DataFrame]:
    return bundle.hiring_summary.copy(), bundle.promotion_summary.copy()


def format_network_table(df: pd.DataFrame, entity: str) -> pd.DataFrame:
    if df.empty:
        return df

    if entity == "manager":
        desired = [
            "scenario",
            "company_id",
            "manager_id",
            "manager_full_name",
            "manager_department_id",
            "headcount",
            "avg_performance",
            "avg_salary",
            "manager_nepotism_risk_score",
        ]
        rename_map = {
            "scenario": "Scenario",
            "company_id": "Company",
            "manager_id": "Manager ID",
            "manager_full_name": "Manager",
            "manager_department_id": "Department",
            "headcount": "Headcount",
            "avg_performance": "Avg performance",
            "avg_salary": "Avg salary",
            "manager_nepotism_risk_score": "Risk score",
        }
    else:
        desired = [
            "scenario",
            "company_id",
            "department_id",
            "headcount",
            "manager_count",
            "avg_performance",
            "avg_salary",
            "department_nepotism_risk_score",
        ]
        rename_map = {
            "scenario": "Scenario",
            "company_id": "Company",
            "department_id": "Department",
            "headcount": "Headcount",
            "manager_count": "Managers",
            "avg_performance": "Avg performance",
            "avg_salary": "Avg salary",
            "department_nepotism_risk_score": "Risk score",
        }

    available = [column for column in desired if column in df.columns]
    formatted = df[available].copy()
    for column in ("avg_performance", "manager_nepotism_risk_score", "department_nepotism_risk_score"):
        if column in formatted.columns:
            formatted[column] = formatted[column].map(lambda value: f"{value:.3f}" if pd.notna(value) else "N/A")
    if "avg_salary" in formatted.columns:
        formatted["avg_salary"] = formatted["avg_salary"].map(lambda value: f"{value:,.0f}" if pd.notna(value) else "N/A")
    return formatted.rename(columns=rename_map)


def format_anomaly_summary_table(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    if df.empty:
        return df

    formatted = df.copy()
    if "anomaly_rate" in formatted.columns:
        formatted["anomaly_rate"] = formatted["anomaly_rate"].map(lambda value: f"{value * 100:.1f}%")
    for column in ("average_anomaly_score", "median_anomaly_score"):
        if column in formatted.columns:
            formatted[column] = formatted[column].map(lambda value: f"{value:.4f}")

    rename_map = {
        "scenario": "Scenario",
        "total_hired": "Total hired",
        "total_promoted": "Total promoted",
        "anomaly_count": "Anomaly count",
        "anomaly_rate": "Anomaly rate",
        "average_anomaly_score": "Avg anomaly score",
        "median_anomaly_score": "Median anomaly score",
    }
    if kind == "hiring":
        desired = ["scenario", "total_hired", "anomaly_count", "anomaly_rate", "average_anomaly_score", "median_anomaly_score"]
    else:
        desired = ["scenario", "total_promoted", "anomaly_count", "anomaly_rate", "average_anomaly_score", "median_anomaly_score"]
    available = [column for column in desired if column in formatted.columns]
    return formatted[available].rename(columns=rename_map)


def format_anomaly_top_table(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    if df.empty:
        return df

    if kind == "hiring":
        desired = [
            "scenario",
            "company_id",
            "employee_id",
            "candidate_id",
            "full_name",
            "merit_score",
            "connection_index",
            "discretionary_channel",
            "hiring_anomaly_score",
            "hiring_anomaly_flag",
        ]
        rename_map = {
            "scenario": "Scenario",
            "company_id": "Company",
            "employee_id": "Employee ID",
            "candidate_id": "Candidate ID",
            "full_name": "Name",
            "merit_score": "Merit score",
            "connection_index": "Connection index",
            "discretionary_channel": "Discretionary channel",
            "hiring_anomaly_score": "Anomaly score",
            "hiring_anomaly_flag": "Flagged",
        }
    else:
        desired = [
            "scenario",
            "company_id",
            "employee_id",
            "full_name",
            "performance_score",
            "connection_index",
            "discretionary_channel",
            "promotion_anomaly_score",
            "promotion_anomaly_flag",
        ]
        rename_map = {
            "scenario": "Scenario",
            "company_id": "Company",
            "employee_id": "Employee ID",
            "full_name": "Name",
            "performance_score": "Performance score",
            "connection_index": "Connection index",
            "discretionary_channel": "Discretionary channel",
            "promotion_anomaly_score": "Anomaly score",
            "promotion_anomaly_flag": "Flagged",
        }

    available = [column for column in desired if column in df.columns]
    formatted = df[available].copy()
    for column in ("merit_score", "performance_score", "connection_index", "hiring_anomaly_score", "promotion_anomaly_score"):
        if column in formatted.columns:
            formatted[column] = formatted[column].map(lambda value: f"{value:.4f}" if pd.notna(value) else "N/A")
    return formatted.rename(columns=rename_map)


def build_bar_series(df: pd.DataFrame, category_column: str, value_column: str) -> pd.DataFrame:
    if df.empty or category_column not in df.columns or value_column not in df.columns:
        return pd.DataFrame()
    out = df[[category_column, value_column]].dropna().copy()
    return out.rename(columns={category_column: "scenario", value_column: "value"})


def _sort_scenarios(df: pd.DataFrame) -> pd.DataFrame:
    ordered = df.copy()
    preferred_order = ["Merit-based", "Moderate favoritism", "High nepotism risk"]
    ordered["scenario"] = pd.Categorical(ordered["scenario"], categories=preferred_order, ordered=True)
    ordered = ordered.sort_values("scenario").reset_index(drop=True)
    ordered["scenario"] = ordered["scenario"].astype(str)
    return ordered
