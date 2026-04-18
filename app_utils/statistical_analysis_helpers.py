from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from typing import Any

import numpy as np
import pandas as pd

from app_utils.input_preprocessing import calculate_hiring_merit_breakdown
from app_utils.model_loader import load_task_bundle


MODULE_PATHS = {
    "hiring": "src.model_hiring",
    "promotion": "src.model_promotion",
}

CONNECTION_COMPONENT_COLUMNS = [
    "family_link_flag",
    "close_family_relation_flag",
    "same_high_school_flag",
    "same_city_flag",
    "same_college_flag",
    "same_last_name_flag",
    "referral_flag",
    "connection_strength",
]

PAIR_SEARCH_SETTINGS = {
    "hiring": [
        {"prob_tol": 0.006, "merit_gap": 10.0, "connection_gap": 0.16},
        {"prob_tol": 0.010, "merit_gap": 8.0, "connection_gap": 0.13},
        {"prob_tol": 0.016, "merit_gap": 6.0, "connection_gap": 0.10},
        {"prob_tol": 0.024, "merit_gap": 4.0, "connection_gap": 0.08},
    ],
    "promotion": [
        {"prob_tol": 0.010, "merit_gap": 10.0, "connection_gap": 0.16},
        {"prob_tol": 0.015, "merit_gap": 8.0, "connection_gap": 0.13},
        {"prob_tol": 0.022, "merit_gap": 6.0, "connection_gap": 0.10},
        {"prob_tol": 0.030, "merit_gap": 4.0, "connection_gap": 0.08},
    ],
}

SENSITIVITY_GRID_SIZE = 15


def _get_module(task: str):
    if task not in MODULE_PATHS:
        raise ValueError(f"Unsupported task: {task}")
    return import_module(MODULE_PATHS[task])


@lru_cache(maxsize=4)
def _load_processed_frame_cached(task: str) -> pd.DataFrame:
    module = _get_module(task)
    path = module.get_paths()["input"]
    return pd.read_csv(path)


def load_processed_frame(task: str) -> pd.DataFrame:
    return _load_processed_frame_cached(task).copy()


def build_connection_index(frame: pd.DataFrame) -> pd.Series:
    available_columns = [column for column in CONNECTION_COMPONENT_COLUMNS if column in frame.columns]
    if not available_columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)

    normalized_columns = []
    for column in available_columns:
        series = pd.to_numeric(frame[column], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        normalized_columns.append(series)

    return pd.concat(normalized_columns, axis=1).mean(axis=1).round(4)


def _predict_probability(task: str, model_name: str, frame: pd.DataFrame) -> np.ndarray:
    module = _get_module(task)
    bundle = load_task_bundle(task)
    live_model = bundle.models[model_name]
    prepared = module.ensure_model_features(frame.copy())
    X = module.transform_features(prepared, live_model.schema)
    return live_model.model.predict_proba(X)[:, 1]


@lru_cache(maxsize=8)
def _score_processed_frame_cached(task: str, model_name: str) -> pd.DataFrame:
    frame = load_processed_frame(task).copy()
    probabilities = _predict_probability(task, model_name, frame)
    frame["predicted_probability"] = probabilities
    frame["connection_index"] = build_connection_index(frame)
    return frame


def score_processed_frame(task: str, model_name: str) -> pd.DataFrame:
    return _score_processed_frame_cached(task, model_name).copy()


def _normalized(series: pd.Series) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce")
    std = float(clean.std(ddof=0))
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=series.index, dtype=float)
    return (clean - float(clean.mean())) / std


def _select_non_overlapping_pairs(pairs: pd.DataFrame, left_id_col: str, right_id_col: str, max_pairs: int) -> pd.DataFrame:
    chosen_rows = []
    used_ids: set[str] = set()
    for row in pairs.itertuples(index=False):
        left_id = str(getattr(row, left_id_col))
        right_id = str(getattr(row, right_id_col))
        if left_id in used_ids or right_id in used_ids:
            continue
        used_ids.update({left_id, right_id})
        chosen_rows.append(row._asdict())
        if len(chosen_rows) >= max_pairs:
            break
    return pd.DataFrame(chosen_rows)


def _pair_search_columns(task: str) -> tuple[str, str]:
    if task == "hiring":
        return "candidate_id", "merit_score"
    return "employee_id", "performance_score"


def _build_scenario_pair_candidates(
    frame: pd.DataFrame,
    *,
    id_column: str,
    merit_column: str,
    setting: dict[str, float],
) -> pd.DataFrame:
    pair_candidates: list[pd.DataFrame] = []
    has_company = "company_id" in frame.columns

    for scenario_name, scenario_df in frame.groupby("scenario", dropna=False):
        scenario_work = scenario_df.dropna(
            subset=["predicted_probability", "connection_index", merit_column]
        ).copy()
        if scenario_work.empty:
            continue

        scenario_work["contrast_score"] = _normalized(scenario_work[merit_column]) - _normalized(
            scenario_work["connection_index"]
        )
        merit_cut = float(scenario_work["contrast_score"].quantile(0.75))
        connection_cut = float(scenario_work["contrast_score"].quantile(0.25))

        merit_pool = scenario_work[scenario_work["contrast_score"] >= merit_cut].copy()
        connection_pool = scenario_work[scenario_work["contrast_score"] <= connection_cut].copy()
        if merit_pool.empty or connection_pool.empty:
            continue

        left = merit_pool.sort_values("predicted_probability").rename(
            columns={
                id_column: "merit_profile_id",
                "predicted_probability": "probability_merit",
                merit_column: "merit_value_merit",
                "connection_index": "connection_index_merit",
                "scenario": "scenario_merit",
                "full_name": "full_name_merit",
                "discretionary_channel": "discretionary_channel_merit",
                "company_id": "company_id_merit",
            }
        )
        right = connection_pool.sort_values("predicted_probability").rename(
            columns={
                id_column: "connection_profile_id",
                "predicted_probability": "probability_connection",
                merit_column: "merit_value_connection",
                "connection_index": "connection_index_connection",
                "scenario": "scenario_connection",
                "full_name": "full_name_connection",
                "discretionary_channel": "discretionary_channel_connection",
                "company_id": "company_id_connection",
            }
        )

        merged = pd.merge_asof(
            left,
            right,
            left_on="probability_merit",
            right_on="probability_connection",
            direction="nearest",
            tolerance=setting["prob_tol"],
        )
        if merged.empty:
            continue

        merged = merged.dropna(subset=["connection_profile_id"]).copy()
        if merged.empty:
            continue

        merged["probability_gap"] = (merged["probability_merit"] - merged["probability_connection"]).abs()
        merged["merit_gap"] = merged["merit_value_merit"] - merged["merit_value_connection"]
        merged["connection_gap"] = merged["connection_index_connection"] - merged["connection_index_merit"]
        merged["same_company_match"] = (
            merged["company_id_merit"].astype(str) == merged["company_id_connection"].astype(str)
            if has_company and {"company_id_merit", "company_id_connection"}.issubset(merged.columns)
            else False
        )
        merged = merged[
            (merged["scenario_merit"].astype(str) == str(scenario_name))
            & (merged["scenario_connection"].astype(str) == str(scenario_name))
            & (merged["merit_gap"] >= setting["merit_gap"])
            & (merged["connection_gap"] >= setting["connection_gap"])
            & (merged["merit_profile_id"].astype(str) != merged["connection_profile_id"].astype(str))
        ].copy()
        if merged.empty:
            continue

        merged["pair_score"] = (
            merged["merit_gap"] * 1.5
            + merged["connection_gap"] * 40.0
            - merged["probability_gap"] * 100.0
            + merged["same_company_match"].astype(int) * 1.0
        )
        pair_candidates.append(merged)

    if not pair_candidates:
        return pd.DataFrame()
    return pd.concat(pair_candidates, ignore_index=True)


@lru_cache(maxsize=8)
def _find_contrasting_pairs_cached(task: str, model_name: str, max_pairs: int = 3) -> dict[str, Any]:
    id_column, merit_column = _pair_search_columns(task)
    scored = score_processed_frame(task, model_name)

    required_columns = {id_column, "scenario", "predicted_probability", "connection_index", merit_column}
    if not required_columns.issubset(scored.columns):
        missing = sorted(required_columns - set(scored.columns))
        return {
            "pairs": pd.DataFrame(),
            "note": f"Required columns missing for pair search: {', '.join(missing)}.",
            "relaxed": False,
        }

    work = scored.dropna(subset=["predicted_probability", "connection_index", merit_column, "scenario"]).copy()
    if work.empty:
        return {"pairs": pd.DataFrame(), "note": "No rows available for pair search.", "relaxed": False}

    for index, setting in enumerate(PAIR_SEARCH_SETTINGS[task]):
        merged = _build_scenario_pair_candidates(
            work,
            id_column=id_column,
            merit_column=merit_column,
            setting=setting,
        )
        if merged.empty:
            continue

        merged = merged.sort_values(
            ["same_company_match", "pair_score", "probability_gap"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
        selected = _select_non_overlapping_pairs(merged, "merit_profile_id", "connection_profile_id", max_pairs)
        if not selected.empty:
            relaxed = index > 0
            note = (
                "Strong same-scenario matched pairs were found under the strict search criteria."
                if not relaxed
                else (
                    "Same-scenario matched-pair criteria were relaxed to find stable within-regime examples with near-equal predicted probabilities. "
                    f"Tolerance used: +/-{setting['prob_tol'] * 100:.1f} percentage points."
                )
            )
            return {"pairs": selected, "note": note, "relaxed": relaxed}

    return {
        "pairs": pd.DataFrame(),
        "note": "No statistically strong same-scenario contrasting pair was found even after relaxing the probability tolerance.",
        "relaxed": True,
    }


def find_contrasting_pairs(task: str, model_name: str, max_pairs: int = 3) -> dict[str, Any]:
    result = _find_contrasting_pairs_cached(task, model_name, max_pairs)
    return {
        "pairs": result["pairs"].copy(),
        "note": result["note"],
        "relaxed": result["relaxed"],
    }


def _representative_numeric_columns(task: str) -> list[str]:
    if task == "hiring":
        return [
            "high_school_gpa",
            "college_gpa",
            "test_score",
            "interview_score",
            "years_experience",
            "connection_strength",
            "merit_score",
        ]
    return [
        "performance_score",
        "tenure_months",
        "role_level",
        "years_experience",
        "salary",
        "connection_strength",
        "merit_score",
    ]


@lru_cache(maxsize=8)
def _get_representative_profile_cached(task: str, model_name: str) -> dict[str, Any]:
    frame = score_processed_frame(task, model_name)
    numeric_columns = [column for column in _representative_numeric_columns(task) if column in frame.columns]
    if not numeric_columns or frame.empty:
        return {}

    medians = frame[numeric_columns].median(numeric_only=True)
    work = frame.copy()
    distance = np.zeros(len(work), dtype=float)
    for column in numeric_columns:
        values = pd.to_numeric(work[column], errors="coerce").fillna(float(medians[column]))
        std = float(values.std(ddof=0)) or 1.0
        distance += ((values - float(medians[column])) / std) ** 2
    work["distance_to_median"] = distance
    row = work.sort_values("distance_to_median").iloc[0]
    return row.to_dict()


def get_representative_profile(task: str, model_name: str) -> dict[str, Any]:
    return dict(_get_representative_profile_cached(task, model_name))


def _build_hiring_profile_from_row(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    values = dict(row)
    if values.get("education_level") == "High School":
        values["college_gpa"] = np.nan
    breakdown = calculate_hiring_merit_breakdown(
        education_level=str(values["education_level"]),
        high_school_gpa=float(values["high_school_gpa"]),
        college_gpa=float(0.0 if pd.isna(values.get("college_gpa")) else values["college_gpa"]),
        test_score=float(values["test_score"]),
        interview_score=float(values["interview_score"]),
        years_experience=float(values["years_experience"]),
    )
    values["merit_score"] = breakdown["merit_score"]
    return values


def _score_single_profile(task: str, model_name: str, profile: dict[str, Any]) -> float:
    frame = pd.DataFrame([profile])
    return float(_predict_probability(task, model_name, frame)[0])


@lru_cache(maxsize=8)
def _build_hiring_merit_sensitivity_cached(task: str, model_name: str) -> dict[str, Any]:
    baseline = get_representative_profile(task, model_name)
    scored = score_processed_frame(task, model_name)
    if not baseline or scored.empty:
        return {"curve": pd.DataFrame(), "note": "Representative baseline profile could not be built."}

    donors = (
        scored.sort_values("merit_score")
        .iloc[np.linspace(0, len(scored) - 1, SENSITIVITY_GRID_SIZE).astype(int)]
        .copy()
    )
    rows = []
    baseline_profile = _build_hiring_profile_from_row(baseline)
    for donor in donors.itertuples(index=False):
        profile = dict(baseline_profile)
        profile.update(
            {
                "education_level": donor.education_level,
                "high_school_gpa": donor.high_school_gpa,
                "college_gpa": donor.college_gpa,
                "test_score": donor.test_score,
                "interview_score": donor.interview_score,
                "years_experience": donor.years_experience,
            }
        )
        profile = _build_hiring_profile_from_row(profile)
        rows.append(
            {
                "x_value": float(profile["merit_score"]),
                "predicted_probability": _score_single_profile(task, model_name, profile),
            }
        )
    curve = pd.DataFrame(rows).drop_duplicates(subset=["x_value"]).sort_values("x_value").reset_index(drop=True)
    note = (
        "Merit-related inputs are varied using real candidate profiles across the observed merit-score range, "
        "while connection-related inputs stay fixed at the representative baseline."
    )
    return {"curve": curve, "note": note, "baseline": baseline}


def build_hiring_merit_sensitivity(task: str, model_name: str) -> dict[str, Any]:
    result = _build_hiring_merit_sensitivity_cached(task, model_name)
    return {
        "curve": result["curve"].copy(),
        "note": result["note"],
        "baseline": dict(result.get("baseline", {})),
    }


@lru_cache(maxsize=8)
def _build_hiring_connection_sensitivity_cached(task: str, model_name: str) -> dict[str, Any]:
    baseline = get_representative_profile(task, model_name)
    if not baseline:
        return {"curve": pd.DataFrame(), "note": "Representative baseline profile could not be built."}

    baseline_profile = _build_hiring_profile_from_row(baseline)
    rows = []
    for value in np.linspace(0.0, 1.0, SENSITIVITY_GRID_SIZE):
        profile = dict(baseline_profile)
        profile["connection_strength"] = float(round(value, 3))
        rows.append(
            {
                "x_value": float(value),
                "predicted_probability": _score_single_profile(task, model_name, profile),
            }
        )
    note = "Connection strength is varied while all merit, flag, and discretionary-channel inputs remain fixed at the representative baseline."
    return {"curve": pd.DataFrame(rows), "note": note, "baseline": baseline}


def build_hiring_connection_sensitivity(task: str, model_name: str) -> dict[str, Any]:
    result = _build_hiring_connection_sensitivity_cached(task, model_name)
    return {
        "curve": result["curve"].copy(),
        "note": result["note"],
        "baseline": dict(result.get("baseline", {})),
    }


@lru_cache(maxsize=8)
def _build_promotion_performance_sensitivity_cached(task: str, model_name: str) -> dict[str, Any]:
    baseline = get_representative_profile(task, model_name)
    scored = score_processed_frame(task, model_name)
    if not baseline or scored.empty:
        return {"curve": pd.DataFrame(), "note": "Representative baseline profile could not be built."}

    lower = float(pd.to_numeric(scored["performance_score"], errors="coerce").quantile(0.05))
    upper = float(pd.to_numeric(scored["performance_score"], errors="coerce").quantile(0.95))
    rows = []
    for value in np.linspace(lower, upper, SENSITIVITY_GRID_SIZE):
        profile = dict(baseline)
        profile["performance_score"] = float(round(value, 2))
        rows.append(
            {
                "x_value": float(value),
                "predicted_probability": _score_single_profile(task, model_name, profile),
            }
        )
    note = "Performance score is varied while tenure, role context, connection features, and discretionary channel remain fixed at the representative baseline."
    return {"curve": pd.DataFrame(rows), "note": note, "baseline": baseline}


def build_promotion_performance_sensitivity(task: str, model_name: str) -> dict[str, Any]:
    result = _build_promotion_performance_sensitivity_cached(task, model_name)
    return {
        "curve": result["curve"].copy(),
        "note": result["note"],
        "baseline": dict(result.get("baseline", {})),
    }


@lru_cache(maxsize=8)
def _build_promotion_connection_sensitivity_cached(task: str, model_name: str) -> dict[str, Any]:
    baseline = get_representative_profile(task, model_name)
    if not baseline:
        return {"curve": pd.DataFrame(), "note": "Representative baseline profile could not be built."}

    rows = []
    for value in np.linspace(0.0, 1.0, SENSITIVITY_GRID_SIZE):
        profile = dict(baseline)
        profile["connection_strength"] = float(round(value, 3))
        rows.append(
            {
                "x_value": float(value),
                "predicted_probability": _score_single_profile(task, model_name, profile),
            }
        )
    note = "Connection strength is varied while performance, tenure, role context, and other connection flags remain fixed at the representative baseline."
    return {"curve": pd.DataFrame(rows), "note": note, "baseline": baseline}


def build_promotion_connection_sensitivity(task: str, model_name: str) -> dict[str, Any]:
    result = _build_promotion_connection_sensitivity_cached(task, model_name)
    return {
        "curve": result["curve"].copy(),
        "note": result["note"],
        "baseline": dict(result.get("baseline", {})),
    }
