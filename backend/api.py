from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
from pydantic import BaseModel, Field

from app_utils.input_preprocessing import (
    build_hiring_input_frame,
    build_promotion_input_frame,
    calculate_hiring_merit_breakdown,
)
from app_utils.data_summary_helpers import (
    build_chart_series,
    build_difference_table,
    compute_overall_average_metrics,
    compute_overview_metrics,
    compute_scenario_summary,
    format_summary_table,
    load_data_summary_bundle_core,
)
from app_utils.dashboard_helpers import (
    build_bar_series,
    compute_anomaly_metrics,
    compute_anomaly_summaries,
    compute_network_metrics,
    compute_network_scenario_averages,
    format_anomaly_summary_table,
    format_anomaly_top_table,
    format_network_table,
    load_dashboard_bundle_core,
)
from app_utils.model_loader import get_model_choices, load_task_bundle
from app_utils.organizational_impact_helpers import (
    format_coefficient_table,
    get_explanatory_inference,
    load_organizational_impact_bundle,
)
from app_utils.prediction_helpers import (
    build_hiring_driver_summary,
    build_logistic_curve_points,
    build_logistic_math_breakdown,
    build_math_equation_lines,
    build_promotion_driver_summary,
    describe_model_version,
    format_probability,
    get_probability_band,
    predict_single_probability,
)
from app_utils.statistical_analysis_helpers import (
    build_hiring_connection_sensitivity,
    build_hiring_merit_sensitivity,
    build_promotion_connection_sensitivity,
    build_promotion_performance_sensitivity,
    find_contrasting_pairs,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_ROOT = PROJECT_ROOT / "frontend"

ROLE_LABELS = {
    1: "Intern / Trainee",
    2: "Entry-Level Staff",
    3: "Senior Staff / Specialist",
    4: "Manager",
    5: "Director / Executive",
}


class HiringPredictionRequest(BaseModel):
    model_name: str | None = None
    education_level: str = Field(default="Bachelor")
    years_experience: float = Field(default=3.0, ge=0.0)
    high_school_gpa: float = Field(default=82.0, ge=50.0, le=100.0)
    college_gpa: float = Field(default=84.0, ge=50.0, le=100.0)
    test_score: float = Field(default=78.0, ge=30.0, le=100.0)
    interview_score: float = Field(default=80.0, ge=30.0, le=100.0)
    connection_strength: float = Field(default=0.2, ge=0.0, le=1.0)
    discretionary_channel: str = Field(default="none")
    referral_flag: bool = False
    family_link_flag: bool = False
    close_family_relation_flag: bool = False
    same_high_school_flag: bool = False
    same_city_flag: bool = False
    same_college_flag: bool = False
    same_last_name_flag: bool = False


class PromotionPredictionRequest(BaseModel):
    model_name: str | None = None
    performance_score: float = Field(default=76.0, ge=35.0, le=100.0)
    tenure_months: float = Field(default=24.0, ge=0.0)
    role_level: int = Field(default=2, ge=1, le=5)
    salary: float = Field(default=32000.0, ge=0.0)
    years_experience: float = Field(default=5.0, ge=0.0)
    merit_score: float = Field(default=74.0, ge=30.0, le=100.0)
    connection_strength: float = Field(default=0.2, ge=0.0, le=1.0)
    discretionary_channel: str = Field(default="none")
    family_link_flag: bool = False
    close_family_relation_flag: bool = False
    same_high_school_flag: bool = False
    same_city_flag: bool = False
    same_college_flag: bool = False
    same_last_name_flag: bool = False


app = FastAPI(
    title="Nepotism Risk Analytics API",
    version="0.1.0",
    description="Backend API for the web migration of Nepotism Risk Analytics.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_ROOT.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_ROOT)), name="static")


def _resolve_model_name(bundle, requested_model_name: str | None) -> str:
    if requested_model_name and requested_model_name in bundle.models:
        return requested_model_name
    return bundle.default_model_name


def _serialize_model_context(bundle, model_name: str) -> dict:
    selected_model = bundle.models[model_name]
    return {
        "model_name": model_name,
        "display_name": selected_model.display_name,
        "description": describe_model_version(selected_model),
        "artifact_status": selected_model.artifact_status,
        "artifact_path": selected_model.artifact_path,
        "source_note": selected_model.source_note,
        "training_rows": selected_model.training_rows,
        "brier_score": selected_model.brier_score,
        "roc_auc": selected_model.roc_auc,
    }


def _serialize_math(bundle, model_name: str, input_df) -> dict:
    math_breakdown = build_logistic_math_breakdown(bundle, model_name, input_df)
    curve_df, marker_df = build_logistic_curve_points(math_breakdown.linear_score)
    contributions = math_breakdown.contributions[
        ["feature", "feature_label", "x_value", "coefficient", "contribution"]
    ].copy()
    return {
        "equation_lines": build_math_equation_lines(math_breakdown),
        "intercept": math_breakdown.intercept,
        "linear_score": math_breakdown.linear_score,
        "raw_probability": math_breakdown.raw_probability,
        "displayed_probability": math_breakdown.displayed_probability,
        "calibration_note": math_breakdown.calibration_note,
        "contributions": contributions.to_dict(orient="records"),
        "curve_points": curve_df.to_dict(orient="records"),
        "marker_points": marker_df.to_dict(orient="records"),
    }


def _clean_json_value(value):
    if isinstance(value, dict):
        return {key: _clean_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean_json_value(item) for item in value]
    if pd.isna(value):
        return None
    return value


def _clean_record_list(records: list[dict]) -> list[dict]:
    return [{key: _clean_json_value(value) for key, value in record.items()} for record in records]


def _serialize_pair_result(result: dict) -> dict:
    pairs = result.get("pairs", pd.DataFrame())
    if isinstance(pairs, pd.DataFrame):
        pair_records = _clean_record_list(pairs.to_dict(orient="records"))
    else:
        pair_records = []
    return {
        "pairs": pair_records,
        "count": len(pair_records),
        "note": result.get("note", ""),
        "relaxed": bool(result.get("relaxed", False)),
    }


def _serialize_sensitivity(result: dict) -> dict:
    curve = result.get("curve", pd.DataFrame())
    baseline = result.get("baseline", {}) or {}
    if isinstance(curve, pd.DataFrame):
        curve_records = _clean_record_list(curve.to_dict(orient="records"))
    else:
        curve_records = []
    return {
        "curve": curve_records,
        "note": result.get("note", ""),
        "baseline": _clean_json_value(baseline),
    }


def _format_percent_value(value: object) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{float(value) * 100:.1f}%"


def _format_number_value(value: object, decimals: int = 2) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{float(value):,.{decimals}f}"


def _format_impact_table_records(
    df: pd.DataFrame,
    *,
    rename_map: dict[str, str],
    percent_columns: list[str],
    decimal_columns: list[str],
    decimal_overrides: dict[str, int] | None = None,
) -> list[dict]:
    if df.empty:
        return []

    formatted = df.rename(columns=rename_map).copy()
    decimal_overrides = decimal_overrides or {}
    for column in percent_columns:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(_format_percent_value)
    for column in decimal_columns:
        if column in formatted.columns:
            decimals = decimal_overrides.get(column, 2)
            formatted[column] = formatted[column].map(lambda value: _format_number_value(value, decimals))
    return _clean_record_list(formatted.to_dict(orient="records"))


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/meta/home")
def get_home_meta() -> dict:
    return {
        "title": "Nepotism Risk Analytics",
        "subtitle": "Backend + frontend migration shell",
        "hero": {
            "title": "Web frontend active",
            "body": (
                "This interface keeps the core modeling logic in Python and moves the presentation layer to a web frontend. "
                "The current migration slice includes Home, Hiring Predictor, Promotion Predictor, Data Summary, "
                "Risk Dashboard, Statistical Analysis, and Organizational Impact."
            ),
        },
        "pipeline": [
            {
                "model": "Model 1",
                "title": "Hiring probability",
                "description": "Candidate-level hiring prediction based on merit, connection indicators, and discretionary channels.",
            },
            {
                "model": "Model 2",
                "title": "Promotion probability",
                "description": "Employee-level promotion prediction based on performance, tenure, role context, and connection indicators.",
            },
            {
                "model": "Model 3",
                "title": "Network risk",
                "description": "Manager and department structural nepotism risk outputs derived from concentration and connectedness patterns.",
            },
            {
                "model": "Model 4",
                "title": "Anomaly scoring",
                "description": "Suspicious hire and promotion anomaly outputs based on merit or performance mismatch and connection intensity.",
            },
        ],
        "migration_note": (
            "The web frontend now covers the live predictor flow plus the analytical inspection pages: Data Summary, "
            "Risk Dashboard, Statistical Analysis, and Organizational Impact."
        ),
    }


@app.get("/api/data-summary")
def get_data_summary() -> dict:
    bundle = load_data_summary_bundle_core()
    scenario_summary = compute_scenario_summary(bundle)
    source_rows = [
        {"dataset": name, "rows": int(len(frame)), "columns": int(len(frame.columns))}
        for name, frame in bundle.frames.items()
    ]

    comparison_specs = [
        ("total_candidates", "Candidates by scenario"),
        ("total_employees", "Employees by scenario"),
        ("hiring_rate_pct", "Hiring rate by scenario"),
        ("promotion_rate_pct", "Promotion rate by scenario"),
        ("avg_candidate_merit_score", "Average candidate merit by scenario"),
        ("avg_candidate_connection_strength", "Average candidate connection by scenario"),
        ("avg_performance_score", "Average employee performance by scenario"),
        ("avg_salary", "Average salary by scenario"),
    ]
    comparisons = {}
    for column, label in comparison_specs:
        series = build_chart_series(scenario_summary, column, label)
        if not series.empty:
            comparisons[column] = series.to_dict(orient="records")

    difference_tables = {}
    for baseline in ["Merit-based", "Moderate favoritism", "High nepotism risk"]:
        diff = build_difference_table(scenario_summary, baseline)
        if not diff.empty:
            difference_tables[baseline] = diff.to_dict(orient="records")

    return {
        "warnings": bundle.warnings,
        "workbook_exists": bundle.workbook_exists,
        "workbook_sheets": bundle.workbook_sheets,
        "source_rows": source_rows,
        "overview_metrics": compute_overview_metrics(bundle),
        "overall_average_metrics": compute_overall_average_metrics(bundle),
        "scenario_summary_raw": scenario_summary.to_dict(orient="records"),
        "scenario_summary_table": format_summary_table(scenario_summary).to_dict(orient="records"),
        "comparisons": comparisons,
        "difference_tables": difference_tables,
    }


@app.get("/api/risk-dashboard")
def get_risk_dashboard() -> dict:
    bundle = load_dashboard_bundle_core()
    network_summary = compute_network_scenario_averages(bundle)
    hiring_summary, promotion_summary = compute_anomaly_summaries(bundle)

    return {
        "warnings": bundle.warnings,
        "notes": bundle.notes,
        "network_metrics": compute_network_metrics(bundle),
        "anomaly_metrics": compute_anomaly_metrics(bundle),
        "network_summary": network_summary.to_dict(orient="records"),
        "manager_risk_series": build_bar_series(network_summary, "scenario", "avg_manager_risk_score").to_dict(orient="records"),
        "department_risk_series": build_bar_series(network_summary, "scenario", "avg_department_risk_score").to_dict(orient="records"),
        "top_managers": format_network_table(bundle.frames.get("top_managers", pd.DataFrame()), "manager").to_dict(orient="records"),
        "top_departments": format_network_table(bundle.frames.get("top_departments", pd.DataFrame()), "department").to_dict(orient="records"),
        "hiring_summary_table": format_anomaly_summary_table(hiring_summary, "hiring").to_dict(orient="records"),
        "promotion_summary_table": format_anomaly_summary_table(promotion_summary, "promotion").to_dict(orient="records"),
        "hiring_anomaly_series": build_bar_series(hiring_summary, "scenario", "anomaly_rate").to_dict(orient="records"),
        "promotion_anomaly_series": build_bar_series(promotion_summary, "scenario", "anomaly_rate").to_dict(orient="records"),
        "top_hiring": format_anomaly_top_table(bundle.frames.get("top_hiring", pd.DataFrame()).head(15), "hiring").to_dict(orient="records"),
        "top_promotion": format_anomaly_top_table(bundle.frames.get("top_promotion", pd.DataFrame()).head(15), "promotion").to_dict(orient="records"),
    }


@app.get("/api/statistical-analysis")
def get_statistical_analysis() -> dict:
    hiring_bundle = load_task_bundle("hiring")
    promotion_bundle = load_task_bundle("promotion")
    hiring_model_name = hiring_bundle.default_model_name
    promotion_model_name = promotion_bundle.default_model_name

    candidate_pairs = find_contrasting_pairs("hiring", hiring_model_name, max_pairs=3)
    employee_pairs = find_contrasting_pairs("promotion", promotion_model_name, max_pairs=3)

    return {
        "title": "Statistical Analysis",
        "subtitle": "Comparative analysis and model-behavior demonstrations",
        "hero": {
            "title": "Understanding merit-connection substitution",
            "body": (
                "This page demonstrates how merit and connection signals trade off in the project's hiring and promotion "
                "models. It reuses the current processed data, persisted predictor bundles, same-scenario matched-pair "
                "logic, and one-variable-at-a-time sensitivity curves from the Python analytics layer."
            ),
        },
        "models": {
            "hiring": _serialize_model_context(hiring_bundle, hiring_model_name),
            "promotion": _serialize_model_context(promotion_bundle, promotion_model_name),
        },
        "candidate_pairs": _serialize_pair_result(candidate_pairs),
        "employee_pairs": _serialize_pair_result(employee_pairs),
        "sensitivity": {
            "hiring_merit": _serialize_sensitivity(build_hiring_merit_sensitivity("hiring", hiring_model_name)),
            "hiring_connection": _serialize_sensitivity(build_hiring_connection_sensitivity("hiring", hiring_model_name)),
            "promotion_performance": _serialize_sensitivity(build_promotion_performance_sensitivity("promotion", promotion_model_name)),
            "promotion_connection": _serialize_sensitivity(build_promotion_connection_sensitivity("promotion", promotion_model_name)),
        },
    }


@app.get("/api/organizational-impact")
def get_organizational_impact() -> dict:
    bundle = load_organizational_impact_bundle()

    executive_raw = bundle.executive_table.copy()
    hiring_quality_raw = bundle.hiring_quality_table.copy()
    promotion_fairness_raw = bundle.promotion_fairness_table.copy()
    efficiency_raw = bundle.efficiency_table.copy()
    structural_risk_raw = bundle.structural_risk_table.copy()

    hiring_inference = get_explanatory_inference("hiring")
    promotion_inference = get_explanatory_inference("promotion")

    efficiency_columns = [
        column for column in [
            "scenario",
            "employee_count",
            "total_proxy_output",
            "proxy_per_employee",
            "proxy_per_100_employees",
            "proxy_per_10000_salary_dollars",
        ]
        if column in efficiency_raw.columns
    ]

    return {
        "title": "Organizational Impact",
        "subtitle": "Scenario-level comparison of hiring quality, promotion fairness, efficiency, and structural risk",
        "hero": {
            "title": "Organizational consequences across HR regimes",
            "body": (
                "This page compares the three HR regimes across hiring quality, promotion fairness, organizational "
                "efficiency, structural risk, and explanatory coefficient inference. It reuses the current processed "
                "datasets and saved model outputs from the Python analytics layer."
            ),
        },
        "warnings": list(bundle.warnings),
        "notes": list(bundle.notes),
        "proxy_column": bundle.proxy_column,
        "proxy_label": bundle.proxy_label,
        "proxy_metric_label": bundle.proxy_metric_label,
        "executive_table": _clean_record_list(executive_raw.to_dict(orient="records")),
        "executive_series": {
            "proxy_per_100_employees": _clean_record_list(build_bar_series(executive_raw, "scenario", "proxy_per_100_employees").to_dict(orient="records")),
            "hiring_anomaly_rate": _clean_record_list(build_bar_series(executive_raw, "scenario", "hiring_anomaly_rate").to_dict(orient="records")),
            "promotion_anomaly_rate": _clean_record_list(build_bar_series(executive_raw, "scenario", "promotion_anomaly_rate").to_dict(orient="records")),
        },
        "hiring_quality_table": _format_impact_table_records(
            hiring_quality_raw,
            rename_map={
                "scenario": "Scenario",
                "avg_hired_merit": "Avg hired merit",
                "share_hires_below_candidate_pool_median": "Hires below candidate-pool median",
                "candidate_pool_avg_merit": "Candidate-pool avg merit",
                "hiring_quality_ratio": "Hiring quality ratio",
                "hiring_anomaly_rate": "Hiring anomaly rate",
            },
            percent_columns=["Hires below candidate-pool median", "Hiring anomaly rate"],
            decimal_columns=["Avg hired merit", "Candidate-pool avg merit", "Hiring quality ratio"],
        ),
        "hiring_quality_series": {
            "avg_hired_merit": _clean_record_list(build_bar_series(hiring_quality_raw, "scenario", "avg_hired_merit").to_dict(orient="records")),
            "hiring_quality_ratio": _clean_record_list(build_bar_series(hiring_quality_raw, "scenario", "hiring_quality_ratio").to_dict(orient="records")),
        },
        "promotion_fairness_table": _format_impact_table_records(
            promotion_fairness_raw,
            rename_map={
                "scenario": "Scenario",
                "promotion_quality_ratio": "Promo quality ratio",
                "share_promoted_below_non_promoted_median": "Promoted below non-promoted median",
                "promoted_outperformed_by_non_promoted_count": "Promoted outperformed by peer",
                "promoted_outperformed_by_non_promoted_rate": "Promoted outperformed by peer rate",
                "top_performer_neglect_rate": "Top-performer neglect rate",
            },
            percent_columns=[
                "Promoted below non-promoted median",
                "Promoted outperformed by peer rate",
                "Top-performer neglect rate",
            ],
            decimal_columns=["Promo quality ratio"],
        ),
        "promotion_fairness_series": {
            "promotion_quality_ratio": _clean_record_list(build_bar_series(promotion_fairness_raw, "scenario", "promotion_quality_ratio").to_dict(orient="records")),
            "top_performer_neglect_rate": _clean_record_list(build_bar_series(promotion_fairness_raw, "scenario", "top_performer_neglect_rate").to_dict(orient="records")),
        },
        "efficiency_table": _format_impact_table_records(
            efficiency_raw[efficiency_columns].copy() if efficiency_columns else pd.DataFrame(),
            rename_map={
                "scenario": "Scenario",
                "employee_count": "Employees",
                "total_proxy_output": f"Total {bundle.proxy_label.lower()}",
                "proxy_per_employee": f"{bundle.proxy_label} per employee",
                "proxy_per_100_employees": bundle.proxy_metric_label,
                "proxy_per_10000_salary_dollars": f"{bundle.proxy_label} per 10,000 salary $",
            },
            percent_columns=[],
            decimal_columns=[
                f"Total {bundle.proxy_label.lower()}",
                f"{bundle.proxy_label} per employee",
                bundle.proxy_metric_label,
                f"{bundle.proxy_label} per 10,000 salary $",
            ],
            decimal_overrides={f"{bundle.proxy_label} per 10,000 salary $": 4},
        ),
        "efficiency_series": {
            "proxy_per_100_employees": _clean_record_list(build_bar_series(efficiency_raw, "scenario", "proxy_per_100_employees").to_dict(orient="records")),
        },
        "structural_risk_table": _format_impact_table_records(
            structural_risk_raw,
            rename_map={
                "scenario": "Scenario",
                "avg_manager_risk_score": "Avg manager risk",
                "avg_department_risk_score": "Avg department risk",
                "hiring_anomaly_rate": "Hiring anomaly rate",
                "promotion_anomaly_rate": "Promotion anomaly rate",
            },
            percent_columns=["Hiring anomaly rate", "Promotion anomaly rate"],
            decimal_columns=["Avg manager risk", "Avg department risk"],
        ),
        "structural_risk_series": {
            "avg_manager_risk_score": _clean_record_list(build_bar_series(structural_risk_raw, "scenario", "avg_manager_risk_score").to_dict(orient="records")),
            "avg_department_risk_score": _clean_record_list(build_bar_series(structural_risk_raw, "scenario", "avg_department_risk_score").to_dict(orient="records")),
        },
        "coefficients": {
            "hiring": {
                "warnings": list(hiring_inference["warnings"]),
                "note": hiring_inference["note"],
                "output_path": hiring_inference["output_path"],
                "table": _clean_record_list(format_coefficient_table(hiring_inference["frame"]).to_dict(orient="records")),
            },
            "promotion": {
                "warnings": list(promotion_inference["warnings"]),
                "note": promotion_inference["note"],
                "output_path": promotion_inference["output_path"],
                "table": _clean_record_list(format_coefficient_table(promotion_inference["frame"]).to_dict(orient="records")),
            },
        },
    }


@app.get("/api/reference/hiring")
def get_hiring_reference() -> dict:
    bundle = load_task_bundle("hiring")
    return {
        "task": "hiring",
        "default_model_name": bundle.default_model_name,
        "model_choices": get_model_choices(bundle),
        "reference_data": bundle.reference_data,
    }


@app.get("/api/reference/promotion")
def get_promotion_reference() -> dict:
    bundle = load_task_bundle("promotion")
    return {
        "task": "promotion",
        "default_model_name": bundle.default_model_name,
        "model_choices": get_model_choices(bundle),
        "reference_data": bundle.reference_data,
        "role_labels": ROLE_LABELS,
    }


@app.post("/api/predict/hiring")
def predict_hiring(payload: HiringPredictionRequest) -> dict:
    bundle = load_task_bundle("hiring")
    model_name = _resolve_model_name(bundle, payload.model_name)
    selected_model = bundle.models[model_name]

    merit_breakdown = calculate_hiring_merit_breakdown(
        education_level=payload.education_level,
        high_school_gpa=payload.high_school_gpa,
        college_gpa=payload.college_gpa,
        test_score=payload.test_score,
        interview_score=payload.interview_score,
        years_experience=payload.years_experience,
    )

    form_values = payload.model_dump()
    form_values["merit_score"] = merit_breakdown["merit_score"]
    input_df = build_hiring_input_frame(form_values)
    probability = predict_single_probability(bundle, model_name, input_df)

    return {
        "task": "hiring",
        "probability": probability,
        "probability_label": format_probability(probability),
        "likelihood_band": get_probability_band(probability),
        "merit_breakdown": merit_breakdown,
        "drivers": build_hiring_driver_summary(form_values, selected_model),
        "model": _serialize_model_context(bundle, model_name),
        "math": _serialize_math(bundle, model_name, input_df),
    }


@app.post("/api/predict/promotion")
def predict_promotion(payload: PromotionPredictionRequest) -> dict:
    bundle = load_task_bundle("promotion")
    model_name = _resolve_model_name(bundle, payload.model_name)
    selected_model = bundle.models[model_name]

    form_values = payload.model_dump()
    input_df = build_promotion_input_frame(form_values)
    probability = predict_single_probability(bundle, model_name, input_df)

    return {
        "task": "promotion",
        "probability": probability,
        "probability_label": format_probability(probability),
        "likelihood_band": get_probability_band(probability),
        "drivers": build_promotion_driver_summary(form_values, selected_model),
        "model": _serialize_model_context(bundle, model_name),
        "math": _serialize_math(bundle, model_name, input_df),
    }


@app.get("/")
def serve_frontend() -> FileResponse:
    index_path = FRONTEND_ROOT / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend entry point not found.")
    return FileResponse(index_path)
