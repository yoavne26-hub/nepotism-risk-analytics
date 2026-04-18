from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd


def _get_module(task: str):
    if task == "hiring":
        from src import model_hiring as module
    else:
        from src import model_promotion as module
    return module


def _flag(value: object) -> int:
    return int(bool(value))


def calculate_hiring_merit_breakdown(
    *,
    education_level: str,
    high_school_gpa: float,
    college_gpa: float,
    test_score: float,
    interview_score: float,
    years_experience: float,
) -> dict[str, float]:
    gpa_component = float(high_school_gpa if education_level == "High School" else college_gpa)
    experience_component = min(float(years_experience), 15.0) / 15.0 * 100.0
    high_school_bonus = 0.005 * float(high_school_gpa)
    merit_score = (
        0.35 * float(test_score)
        + 0.33 * float(interview_score)
        + 0.20 * gpa_component
        + 0.12 * experience_component
        + high_school_bonus
    )
    return {
        "gpa_component": gpa_component,
        "experience_component": experience_component,
        "high_school_bonus": high_school_bonus,
        "merit_score": float(np.clip(round(merit_score, 1), 35.0, 100.0)),
    }


def calculate_hiring_merit_score(
    *,
    education_level: str,
    high_school_gpa: float,
    college_gpa: float,
    test_score: float,
    interview_score: float,
    years_experience: float,
) -> float:
    breakdown = calculate_hiring_merit_breakdown(
        education_level=education_level,
        high_school_gpa=high_school_gpa,
        college_gpa=college_gpa,
        test_score=test_score,
        interview_score=interview_score,
        years_experience=years_experience,
    )
    return float(breakdown["merit_score"])


def build_hiring_input_frame(values: Mapping[str, object]) -> pd.DataFrame:
    module = _get_module("hiring")
    college_gpa = values["college_gpa"]
    if values["education_level"] == "High School":
        college_gpa = np.nan

    merit_score = calculate_hiring_merit_score(
        education_level=str(values["education_level"]),
        high_school_gpa=float(values["high_school_gpa"]),
        college_gpa=float(values["college_gpa"]),
        test_score=float(values["test_score"]),
        interview_score=float(values["interview_score"]),
        years_experience=float(values["years_experience"]),
    )

    frame = pd.DataFrame(
        [
            {
                "education_level": values["education_level"],
                "high_school_gpa": float(values["high_school_gpa"]),
                "college_gpa": college_gpa,
                "test_score": float(values["test_score"]),
                "interview_score": float(values["interview_score"]),
                "years_experience": float(values["years_experience"]),
                "merit_score": merit_score,
                "referral_flag": _flag(values["referral_flag"]),
                "family_link_flag": _flag(values["family_link_flag"]),
                "close_family_relation_flag": _flag(values["close_family_relation_flag"]),
                "same_high_school_flag": _flag(values["same_high_school_flag"]),
                "same_city_flag": _flag(values["same_city_flag"]),
                "same_college_flag": _flag(values["same_college_flag"]),
                "same_last_name_flag": _flag(values["same_last_name_flag"]),
                "connection_strength": float(values["connection_strength"]),
                "discretionary_channel": str(values["discretionary_channel"]),
                "hired_flag": 0,
            }
        ]
    )
    return module.ensure_model_features(frame)


def build_promotion_input_frame(values: Mapping[str, object]) -> pd.DataFrame:
    module = _get_module("promotion")
    frame = pd.DataFrame(
        [
            {
                "performance_score": float(values["performance_score"]),
                "tenure_months": float(values["tenure_months"]),
                "role_level": int(values["role_level"]),
                "salary": float(values["salary"]),
                "years_experience": float(values["years_experience"]),
                "merit_score": float(values["merit_score"]),
                "family_link_flag": _flag(values["family_link_flag"]),
                "close_family_relation_flag": _flag(values["close_family_relation_flag"]),
                "same_high_school_flag": _flag(values["same_high_school_flag"]),
                "same_city_flag": _flag(values["same_city_flag"]),
                "same_college_flag": _flag(values["same_college_flag"]),
                "same_last_name_flag": _flag(values["same_last_name_flag"]),
                "connection_strength": float(values["connection_strength"]),
                "discretionary_channel": str(values["discretionary_channel"]),
                "promoted_flag": 0,
            }
        ]
    )
    return module.ensure_model_features(frame)
