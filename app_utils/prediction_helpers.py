from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from app_utils.model_loader import LiveModel, TaskBundle


@dataclass
class LogisticMathBreakdown:
    transformed_row: pd.DataFrame
    contributions: pd.DataFrame
    intercept: float
    linear_score: float
    raw_probability: float
    displayed_probability: float
    calibration_note: str | None


def _get_module(task: str):
    if task == "hiring":
        from src import model_hiring as module
    else:
        from src import model_promotion as module
    return module


def _sigmoid(value: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))


def _friendly_feature_name(feature: str) -> str:
    explicit_names = {
        "education_level_score": "Education level score",
        "promotion_headroom": "Promotion headroom",
        "has_discretionary_channel": "Has discretionary channel",
        "discretionary_family_flag": "Discretionary family channel",
        "discretionary_referral_flag": "Discretionary referral channel",
        "discretionary_manager_endorsement_flag": "Manager endorsement channel",
        "discretionary_alumni_network_flag": "Alumni network channel",
        "discretionary_legacy_referral_flag": "Legacy referral channel",
        "discretionary_internal_source_flag": "Internal source channel",
        "family_link_flag": "Family link flag",
        "close_family_relation_flag": "Close family relation flag",
        "same_high_school_flag": "Same high school flag",
        "same_city_flag": "Same city flag",
        "same_college_flag": "Same college flag",
        "same_last_name_flag": "Same last name flag",
        "referral_flag": "Referral flag",
        "connection_strength": "Connection strength",
    }
    if feature in explicit_names:
        return explicit_names[feature]
    if feature.startswith("discretionary_channel_"):
        return "Discretionary channel: " + feature.replace("discretionary_channel_", "").replace("_", " ")
    if "_x_discretionary_channel_" in feature:
        left, right = feature.split("_x_discretionary_channel_", maxsplit=1)
        return f"{left.replace('_', ' ')} x channel {right.replace('_', ' ')}"
    return feature.replace("_x_", " x ").replace("_", " ")


def _extract_logistic_coefficients(live_model: LiveModel) -> tuple[np.ndarray, float, str | None]:
    model = live_model.model

    if hasattr(model, "coef_") and hasattr(model, "intercept_"):
        return model.coef_.ravel(), float(model.intercept_[0]), None

    if hasattr(model, "calibrated_classifiers_"):
        estimators = [
            calibrated.estimator
            for calibrated in model.calibrated_classifiers_
            if hasattr(calibrated, "estimator") and hasattr(calibrated.estimator, "coef_")
        ]
        if estimators:
            coefficients = np.mean([estimator.coef_.ravel() for estimator in estimators], axis=0)
            intercept = float(np.mean([estimator.intercept_[0] for estimator in estimators]))
            note = (
                "Displayed probability comes from the calibrated model output. The coefficient view below uses the "
                "average internal logistic coefficients before the calibration layer is applied."
            )
            return coefficients, intercept, note

    raise ValueError("The selected model does not expose a logistic coefficient view.")


def predict_single_probability(bundle: TaskBundle, model_name: str, input_df: pd.DataFrame) -> float:
    live_model = bundle.models[model_name]
    module = _get_module(bundle.task)
    X = module.transform_features(input_df, live_model.schema)
    probability = float(live_model.model.predict_proba(X)[:, 1][0])
    return probability


def build_logistic_math_breakdown(bundle: TaskBundle, model_name: str, input_df: pd.DataFrame) -> LogisticMathBreakdown:
    live_model = bundle.models[model_name]
    module = _get_module(bundle.task)
    transformed_row = module.transform_features(input_df, live_model.schema).copy()
    coefficients, intercept, calibration_note = _extract_logistic_coefficients(live_model)

    values = transformed_row.iloc[0].astype(float).to_numpy()
    contributions = pd.DataFrame(
        {
            "feature": transformed_row.columns,
            "feature_label": [_friendly_feature_name(column) for column in transformed_row.columns],
            "x_value": values,
            "coefficient": coefficients,
        }
    )
    contributions["contribution"] = contributions["x_value"] * contributions["coefficient"]
    contributions["abs_contribution"] = contributions["contribution"].abs()
    contributions = contributions.sort_values("abs_contribution", ascending=False).reset_index(drop=True)

    linear_score = float(intercept + contributions["contribution"].sum())
    raw_probability = float(_sigmoid(linear_score))
    displayed_probability = float(live_model.model.predict_proba(transformed_row)[:, 1][0])

    return LogisticMathBreakdown(
        transformed_row=transformed_row,
        contributions=contributions,
        intercept=intercept,
        linear_score=linear_score,
        raw_probability=raw_probability,
        displayed_probability=displayed_probability,
        calibration_note=calibration_note,
    )


def build_logistic_curve_points(linear_score: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    min_score = min(-8.0, linear_score - 2.0)
    max_score = max(8.0, linear_score + 2.0)
    curve_scores = np.linspace(min_score, max_score, 241)
    curve_df = pd.DataFrame(
        {
            "linear_score": curve_scores,
            "probability": _sigmoid(curve_scores),
        }
    )
    marker_df = pd.DataFrame(
        {
            "linear_score": [linear_score],
            "probability": [float(_sigmoid(linear_score))],
            "label": ["Current input"],
        }
    )
    return curve_df, marker_df


def build_math_equation_lines(math_breakdown: LogisticMathBreakdown, top_terms: int = 10) -> list[str]:
    lines = [f"z = {math_breakdown.intercept:.4f}"]
    preview = math_breakdown.contributions.head(top_terms)
    for row in preview.itertuples(index=False):
        lines.append(
            f"  + ({row.coefficient:.4f} x {row.x_value:.4f})  [{row.feature}] = {row.contribution:.4f}"
        )
    remaining_terms = max(len(math_breakdown.contributions) - len(preview), 0)
    if remaining_terms:
        lines.append(f"  + ... {remaining_terms} additional feature terms omitted in this preview")
    lines.append(f"z = {math_breakdown.linear_score:.4f}")
    lines.append(f"p_raw = 1 / (1 + exp(-z)) = {math_breakdown.raw_probability:.4f}")
    if math_breakdown.calibration_note:
        lines.append(f"p_displayed = calibrated(p_raw) = {math_breakdown.displayed_probability:.4f}")
    else:
        lines.append(f"p_displayed = p_raw = {math_breakdown.displayed_probability:.4f}")
    return lines


def format_probability(probability: float) -> str:
    return f"{probability * 100:.1f}%"


def get_probability_band(probability: float) -> str:
    if probability >= 0.70:
        return "High"
    if probability >= 0.40:
        return "Moderate"
    return "Low"


def describe_model_version(live_model: LiveModel) -> str:
    calibrated = "calibrated" if live_model.calibrated else "uncalibrated"
    weight_text = live_model.class_weight or "unweighted"
    return f"{live_model.display_name} | {weight_text} | {calibrated}"


def _flag_sum(values: list[bool | int]) -> int:
    return int(sum(bool(value) for value in values))


def _average(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(values))


def build_hiring_driver_summary(form_values: Mapping[str, object], live_model: LiveModel) -> list[str]:
    high_school_gpa = float(form_values["high_school_gpa"])
    test_score = float(form_values["test_score"])
    interview_score = float(form_values["interview_score"])
    merit_score = float(form_values["merit_score"])
    years_experience = float(form_values["years_experience"])
    connection_strength = float(form_values["connection_strength"])
    education_level = str(form_values["education_level"])
    discretionary_channel = str(form_values["discretionary_channel"])

    academic_values = [high_school_gpa, test_score, interview_score]
    if education_level != "High School":
        academic_values.append(float(form_values["college_gpa"]))
    academic_average = _average(academic_values)

    connection_flags = _flag_sum(
        [
            form_values["referral_flag"],
            form_values["family_link_flag"],
            form_values["close_family_relation_flag"],
            form_values["same_high_school_flag"],
            form_values["same_city_flag"],
            form_values["same_college_flag"],
            form_values["same_last_name_flag"],
        ]
    )

    messages: list[tuple[float, str]] = []
    if academic_average >= 85 and merit_score >= 80:
        messages.append((0.95, "High academic and interview/test results strengthen the merit profile."))
    elif academic_average <= 68 or merit_score <= 60:
        messages.append((0.95, "Lower academic or interview/test scores weaken the merit profile."))

    if years_experience >= 5:
        messages.append((0.70, "Solid prior experience supports the hiring case."))
    elif years_experience < 1:
        messages.append((0.55, "Very limited prior experience reduces the experience signal."))

    if education_level in {"Bachelor", "Master", "Doctorate"}:
        messages.append((0.52, "Post-secondary education satisfies the model's education requirement logic."))
    elif years_experience < 3:
        messages.append((0.52, "High-school-only profile with limited experience leaves a weaker qualification signal."))

    if connection_strength >= 0.60 or connection_flags >= 2:
        messages.append((0.90, "Strong connection indicators materially increase the connection-related signal."))
    elif connection_strength <= 0.15 and connection_flags == 0:
        messages.append((0.70, "Very weak connection indicators keep the profile closer to merit-only evaluation."))

    if discretionary_channel != "none":
        messages.append((0.62, f"The selected discretionary channel ({discretionary_channel.replace('_', ' ')}) adds a non-standard entry path."))

    ranked = [message for _, message in sorted(messages, key=lambda item: item[0], reverse=True)]
    if not ranked:
        ranked = ["The entered profile is fairly neutral across both merit and connection signals."]
    return ranked[:3]


def build_promotion_driver_summary(form_values: Mapping[str, object], live_model: LiveModel) -> list[str]:
    performance_score = float(form_values["performance_score"])
    tenure_months = float(form_values["tenure_months"])
    role_level = int(form_values["role_level"])
    merit_score = float(form_values["merit_score"])
    connection_strength = float(form_values["connection_strength"])
    discretionary_channel = str(form_values["discretionary_channel"])

    connection_flags = _flag_sum(
        [
            form_values["family_link_flag"],
            form_values["close_family_relation_flag"],
            form_values["same_high_school_flag"],
            form_values["same_city_flag"],
            form_values["same_college_flag"],
            form_values["same_last_name_flag"],
        ]
    )

    messages: list[tuple[float, str]] = []
    if performance_score >= 80:
        messages.append((0.92, "Strong performance score supports promotion likelihood."))
    elif performance_score <= 60:
        messages.append((0.92, "Low performance score makes promotion less likely on the merit side."))

    if tenure_months >= 36:
        messages.append((0.72, "Longer tenure strengthens the promotion-readiness signal."))
    elif tenure_months < 12:
        messages.append((0.60, "Short tenure weakens the promotion-readiness signal."))

    if role_level >= 4:
        messages.append((0.64, "A higher current role level may leave less room for additional promotion."))
    elif role_level <= 2:
        messages.append((0.40, "A lower current role level leaves more room for upward movement."))

    if connection_strength >= 0.60 or connection_flags >= 2:
        messages.append((0.88, "Strong connection indicators materially increase the connection-related signal."))
    elif connection_strength <= 0.15 and connection_flags == 0:
        messages.append((0.68, "Weak connection indicators keep the profile closer to performance and tenure alone."))

    if discretionary_channel != "none":
        messages.append((0.58, f"The selected discretionary channel ({discretionary_channel.replace('_', ' ')}) adds a discretionary promotion signal."))

    ranked = [message for _, message in sorted(messages, key=lambda item: item[0], reverse=True)]
    if not ranked:
        ranked = ["The entered profile is fairly neutral across performance, tenure, and connection signals."]
    return ranked[:3]
