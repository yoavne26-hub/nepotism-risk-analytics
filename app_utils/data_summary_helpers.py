from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATHS = {
    "generated_workbook": PROJECT_ROOT / "data" / "generated" / "nepotism_synthetic_data.xlsx",
    "candidates": PROJECT_ROOT / "data" / "processed" / "candidates_model.csv",
    "employees": PROJECT_ROOT / "data" / "processed" / "employees_model.csv",
    "managers": PROJECT_ROOT / "data" / "processed" / "manager_features.csv",
    "departments": PROJECT_ROOT / "data" / "processed" / "department_features.csv",
}


@dataclass
class DataSummaryBundle:
    workbook_exists: bool
    workbook_sheets: list[str]
    frames: dict[str, pd.DataFrame]
    warnings: list[str]
    overview_metrics: list[dict[str, Any]]
    overall_average_metrics: list[dict[str, str]]
    scenario_summary: pd.DataFrame


def _safe_read_csv(path: Path, label: str, warnings: list[str]) -> pd.DataFrame | None:
    if not path.exists():
        warnings.append(f"Missing file: `{path.relative_to(PROJECT_ROOT)}`")
        return None

    try:
        return pd.read_csv(path)
    except Exception as exc:
        warnings.append(f"Could not read `{path.relative_to(PROJECT_ROOT)}`: {exc}")
        return None


def _safe_workbook_sheet_names(path: Path, warnings: list[str]) -> tuple[bool, list[str]]:
    if not path.exists():
        warnings.append(f"Missing file: `{path.relative_to(PROJECT_ROOT)}`")
        return False, []

    try:
        workbook = pd.ExcelFile(path)
        return True, workbook.sheet_names
    except Exception as exc:
        warnings.append(f"Could not inspect `{path.relative_to(PROJECT_ROOT)}`: {exc}")
        return False, []


@lru_cache(maxsize=1)
def load_data_summary_bundle_core() -> DataSummaryBundle:
    warnings: list[str] = []
    workbook_exists, workbook_sheets = _safe_workbook_sheet_names(DATA_PATHS["generated_workbook"], warnings)

    frames: dict[str, pd.DataFrame] = {}
    for key in ("candidates", "employees", "managers", "departments"):
        frame = _safe_read_csv(DATA_PATHS[key], key, warnings)
        if frame is not None:
            frames[key] = frame

    return DataSummaryBundle(
        workbook_exists=workbook_exists,
        workbook_sheets=workbook_sheets,
        frames=frames,
        warnings=warnings,
        overview_metrics=_compute_overview_metrics(frames),
        overall_average_metrics=_compute_overall_average_metrics(frames),
        scenario_summary=_compute_scenario_summary(frames),
    )


def load_data_summary_bundle() -> DataSummaryBundle:
    return load_data_summary_bundle_core()


def _compute_overview_metrics(frames: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    candidates = frames.get("candidates")
    employees = frames.get("employees")
    managers = frames.get("managers")
    departments = frames.get("departments")

    metrics: list[dict[str, Any]] = []

    if candidates is not None:
        metrics.append({"label": "Total candidates", "value": f"{len(candidates):,}"})
        if "company_id" in candidates.columns:
            metrics.append({"label": "Number of companies", "value": f"{candidates['company_id'].nunique():,}"})
        if "scenario" in candidates.columns:
            metrics.append({"label": "Number of scenarios", "value": f"{candidates['scenario'].nunique():,}"})
        if "hired_flag" in candidates.columns:
            metrics.append({"label": "Overall hiring rate", "value": f"{candidates['hired_flag'].mean() * 100:.1f}%"})

    if employees is not None:
        metrics.append({"label": "Total employees", "value": f"{len(employees):,}"})
        if "promoted_flag" in employees.columns:
            metrics.append({"label": "Overall promotion rate", "value": f"{employees['promoted_flag'].mean() * 100:.1f}%"})

    if managers is not None:
        metrics.append({"label": "Total managers", "value": f"{len(managers):,}"})

    if departments is not None:
        metrics.append({"label": "Total departments", "value": f"{len(departments):,}"})

    return metrics


def compute_overview_metrics(bundle: DataSummaryBundle) -> list[dict[str, Any]]:
    return list(bundle.overview_metrics)


def _compute_scenario_summary(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    scenario_summary: pd.DataFrame | None = None

    candidates = frames.get("candidates")
    if candidates is not None and "scenario" in candidates.columns:
        candidate_group = candidates.groupby("scenario", dropna=False)
        candidate_summary = pd.DataFrame({"scenario": list(candidate_group.size().index)})
        candidate_summary["total_candidates"] = candidate_group.size().values

        candidate_metric_map = {
            "hired_flag": ("hiring_rate", "mean"),
            "merit_score": ("avg_candidate_merit_score", "mean"),
            "connection_strength": ("avg_candidate_connection_strength", "mean"),
            "family_link_flag": ("candidate_family_link_rate", "mean"),
            "referral_flag": ("candidate_referral_rate", "mean"),
        }
        for source_column, (output_column, agg_name) in candidate_metric_map.items():
            if source_column in candidates.columns:
                candidate_summary[output_column] = getattr(candidate_group[source_column], agg_name)().values

        scenario_summary = candidate_summary

    employees = frames.get("employees")
    if employees is not None and "scenario" in employees.columns:
        employee_group = employees.groupby("scenario", dropna=False)
        employee_summary = pd.DataFrame({"scenario": list(employee_group.size().index)})
        employee_summary["total_employees"] = employee_group.size().values

        employee_metric_map = {
            "promoted_flag": ("promotion_rate", "mean"),
            "merit_score": ("avg_employee_merit_score", "mean"),
            "connection_strength": ("avg_employee_connection_strength", "mean"),
            "performance_score": ("avg_performance_score", "mean"),
            "salary": ("avg_salary", "mean"),
            "family_link_flag": ("employee_family_link_rate", "mean"),
            "referral_flag": ("employee_referral_rate", "mean"),
        }
        for source_column, (output_column, agg_name) in employee_metric_map.items():
            if source_column in employees.columns:
                employee_summary[output_column] = getattr(employee_group[source_column], agg_name)().values

        if scenario_summary is None:
            scenario_summary = employee_summary
        else:
            scenario_summary = scenario_summary.merge(employee_summary, on="scenario", how="outer")

    if scenario_summary is None:
        return pd.DataFrame()

    if "hiring_rate" in scenario_summary.columns:
        scenario_summary["hiring_rate_pct"] = scenario_summary["hiring_rate"] * 100.0
    if "promotion_rate" in scenario_summary.columns:
        scenario_summary["promotion_rate_pct"] = scenario_summary["promotion_rate"] * 100.0
    if "candidate_family_link_rate" in scenario_summary.columns:
        scenario_summary["candidate_family_link_rate_pct"] = scenario_summary["candidate_family_link_rate"] * 100.0
    if "candidate_referral_rate" in scenario_summary.columns:
        scenario_summary["candidate_referral_rate_pct"] = scenario_summary["candidate_referral_rate"] * 100.0
    if "employee_family_link_rate" in scenario_summary.columns:
        scenario_summary["employee_family_link_rate_pct"] = scenario_summary["employee_family_link_rate"] * 100.0
    if "employee_referral_rate" in scenario_summary.columns:
        scenario_summary["employee_referral_rate_pct"] = scenario_summary["employee_referral_rate"] * 100.0

    preferred_order = [
        "Merit-based",
        "Moderate favoritism",
        "High nepotism risk",
    ]
    scenario_summary["scenario"] = pd.Categorical(
        scenario_summary["scenario"],
        categories=preferred_order,
        ordered=True,
    )
    scenario_summary = scenario_summary.sort_values("scenario").reset_index(drop=True)
    scenario_summary["scenario"] = scenario_summary["scenario"].astype(str)
    return scenario_summary


def compute_scenario_summary(bundle: DataSummaryBundle) -> pd.DataFrame:
    return bundle.scenario_summary.copy()


def _compute_overall_average_metrics(frames: dict[str, pd.DataFrame]) -> list[dict[str, str]]:
    candidates = frames.get("candidates")
    employees = frames.get("employees")

    metrics: list[dict[str, str]] = []
    if candidates is not None:
        if "merit_score" in candidates.columns:
            metrics.append(
                {
                    "label": "Overall candidate merit",
                    "value": f"{candidates['merit_score'].mean():.2f}",
                }
            )
        if "connection_strength" in candidates.columns:
            metrics.append(
                {
                    "label": "Overall candidate connection",
                    "value": f"{candidates['connection_strength'].mean():.2f}",
                }
            )

    if employees is not None:
        if "performance_score" in employees.columns:
            metrics.append(
                {
                    "label": "Overall employee performance",
                    "value": f"{employees['performance_score'].mean():.2f}",
                }
            )
        if "salary" in employees.columns:
            metrics.append(
                {
                    "label": "Overall average salary",
                    "value": f"{employees['salary'].mean():,.0f}",
                }
            )

    return metrics


def compute_overall_average_metrics(bundle: DataSummaryBundle) -> list[dict[str, str]]:
    return list(bundle.overall_average_metrics)


def build_chart_series(summary_df: pd.DataFrame, value_column: str, label: str) -> pd.DataFrame:
    if summary_df.empty or "scenario" not in summary_df.columns or value_column not in summary_df.columns:
        return pd.DataFrame()

    chart_df = summary_df[["scenario", value_column]].copy()
    chart_df = chart_df.dropna()
    chart_df = chart_df.rename(columns={value_column: "value"})
    chart_df["metric"] = label
    return chart_df


def build_difference_table(summary_df: pd.DataFrame, baseline_scenario: str) -> pd.DataFrame:
    if summary_df.empty or "scenario" not in summary_df.columns:
        return pd.DataFrame()

    baseline_rows = summary_df[summary_df["scenario"] == baseline_scenario]
    if baseline_rows.empty:
        return pd.DataFrame()

    baseline_row = baseline_rows.iloc[0]
    metrics = [
        ("total_candidates", "Candidates", "{:,.0f}", "{:+,.0f}"),
        ("total_employees", "Employees", "{:,.0f}", "{:+,.0f}"),
        ("hiring_rate_pct", "Hiring rate (%)", "{:.2f}", "{:+.2f}"),
        ("promotion_rate_pct", "Promotion rate (%)", "{:.2f}", "{:+.2f}"),
        ("avg_candidate_merit_score", "Candidate merit", "{:.2f}", "{:+.2f}"),
        ("avg_candidate_connection_strength", "Candidate connection", "{:.2f}", "{:+.2f}"),
        ("avg_performance_score", "Employee performance", "{:.2f}", "{:+.2f}"),
        ("avg_salary", "Average salary", "{:,.0f}", "{:+,.0f}"),
    ]

    rows: list[dict[str, str]] = []
    for _, row in summary_df.iterrows():
        comparison_row: dict[str, str] = {"Scenario": str(row["scenario"])}
        for column, label, pattern, delta_pattern in metrics:
            if column not in summary_df.columns or pd.isna(row.get(column)) or pd.isna(baseline_row.get(column)):
                continue
            current_value = float(row[column])
            baseline_value = float(baseline_row[column])
            delta_value = current_value - baseline_value
            comparison_row[label] = pattern.format(current_value)
            comparison_row[f"{label} vs {baseline_scenario}"] = delta_pattern.format(delta_value)
        rows.append(comparison_row)

    return pd.DataFrame(rows)


def format_summary_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df

    formatted = summary_df.copy()
    percentage_columns = [
        column
        for column in formatted.columns
        if column.endswith("_pct")
    ]
    average_columns = [
        "avg_candidate_merit_score",
        "avg_candidate_connection_strength",
        "avg_employee_merit_score",
        "avg_employee_connection_strength",
        "avg_performance_score",
        "avg_salary",
    ]

    for column in percentage_columns:
        formatted[column] = formatted[column].map(lambda value: f"{value:.1f}%" if pd.notna(value) else "N/A")

    for column in average_columns:
        if column not in formatted.columns:
            continue
        if column == "avg_salary":
            formatted[column] = formatted[column].map(lambda value: f"{value:,.0f}" if pd.notna(value) else "N/A")
        else:
            formatted[column] = formatted[column].map(lambda value: f"{value:.2f}" if pd.notna(value) else "N/A")

    rename_map = {
        "scenario": "Scenario",
        "total_candidates": "Candidates",
        "total_employees": "Employees",
        "hiring_rate_pct": "Hiring rate",
        "promotion_rate_pct": "Promotion rate",
        "avg_candidate_merit_score": "Avg candidate merit",
        "avg_candidate_connection_strength": "Avg candidate connection",
        "avg_performance_score": "Avg performance",
        "avg_salary": "Avg salary",
        "candidate_family_link_rate_pct": "Candidate family-link rate",
        "candidate_referral_rate_pct": "Candidate referral rate",
        "employee_family_link_rate_pct": "Employee family-link rate",
        "employee_referral_rate_pct": "Employee referral rate",
    }

    desired_columns = [
        "scenario",
        "total_candidates",
        "total_employees",
        "hiring_rate_pct",
        "promotion_rate_pct",
        "avg_candidate_merit_score",
        "avg_candidate_connection_strength",
        "avg_performance_score",
        "avg_salary",
        "candidate_family_link_rate_pct",
        "candidate_referral_rate_pct",
        "employee_family_link_rate_pct",
        "employee_referral_rate_pct",
    ]
    available_columns = [column for column in desired_columns if column in formatted.columns]
    return formatted[available_columns].rename(columns=rename_map)
