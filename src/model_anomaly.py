from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TOP_OUTPUT_ROWS = 50
HIRING_THRESHOLD_QUANTILE = 0.99
PROMOTION_THRESHOLD_QUANTILE = 0.99


def get_paths() -> dict[str, Path]:
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "outputs" / "anomaly_model"
    return {
        "candidates": project_root / "data" / "processed" / "candidates_model.csv",
        "employees": project_root / "data" / "processed" / "employees_model.csv",
        "output_dir": output_dir,
        "hiring_scores": output_dir / "hiring_anomaly_scores.csv",
        "promotion_scores": output_dir / "promotion_anomaly_scores.csv",
        "top_hiring": output_dir / "top_hiring_anomalies.csv",
        "top_promotion": output_dir / "top_promotion_anomalies.csv",
        "hiring_summary": output_dir / "hiring_anomaly_rate_by_scenario.csv",
        "promotion_summary": output_dir / "promotion_anomaly_rate_by_scenario.csv",
        "hiring_chart": output_dir / "hiring_anomaly_scores.png",
        "promotion_chart": output_dir / "promotion_anomaly_scores.png",
    }


def load_inputs(paths: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = pd.read_csv(paths["candidates"])
    employees = pd.read_csv(paths["employees"])
    print(f"Loaded {len(candidates)} candidate rows from {paths['candidates']}")
    print(f"Loaded {len(employees)} employee rows from {paths['employees']}")
    return candidates, employees


def normalize_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").astype(float)
    minimum = float(numeric.min())
    maximum = float(numeric.max())
    if np.isclose(minimum, maximum):
        return pd.Series(np.zeros(len(numeric)), index=series.index, dtype=float)
    return (numeric - minimum) / (maximum - minimum)


def build_connection_index(df: pd.DataFrame) -> pd.Series:
    weights = {
        "close_family_relation_flag": 0.24,
        "family_link_flag": 0.20,
        "same_high_school_flag": 0.15,
        "same_city_flag": 0.10,
        "same_college_flag": 0.09,
        "same_last_name_flag": 0.04,
        "referral_flag": 0.08,
        "connection_strength": 0.10,
    }

    score = pd.Series(np.zeros(len(df)), index=df.index, dtype=float)
    for column, weight in weights.items():
        if column not in df.columns:
            values = pd.Series(np.zeros(len(df)), index=df.index, dtype=float)
        else:
            values = pd.to_numeric(df[column], errors="coerce").fillna(0.0).astype(float)
        if column != "connection_strength":
            values = values.clip(0.0, 1.0)
        else:
            values = values.clip(0.0, 1.0)
        score = score + weight * values

    max_weight_sum = sum(weights.values())
    return (score / max_weight_sum).clip(0.0, 1.0).round(4)


def build_discretion_factor(series: pd.Series) -> pd.Series:
    normalized = series.fillna("none").astype(str).str.strip().str.lower()
    return pd.Series(np.where(normalized != "none", 1.25, 1.0), index=series.index, dtype=float)


def build_hiring_anomaly_frame(candidates: pd.DataFrame, employees: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    candidate_lookup = candidates[
        [
            "candidate_id",
            "company_id",
            "scenario",
            "full_name",
            "merit_score",
            "close_family_relation_flag",
            "family_link_flag",
            "same_high_school_flag",
            "same_city_flag",
            "same_college_flag",
            "same_last_name_flag",
            "referral_flag",
            "connection_strength",
            "discretionary_channel",
        ]
    ].copy()

    hired = employees[
        ["employee_id", "candidate_id", "company_id", "scenario", "full_name"]
    ].copy()
    hired["hired_flag"] = 1

    hiring_df = hired.merge(
        candidate_lookup,
        on=["candidate_id", "company_id", "scenario"],
        how="left",
        suffixes=("_employee", ""),
    )
    hiring_df["full_name"] = hiring_df["full_name"].fillna(hiring_df["full_name_employee"])
    hiring_df["merit_score"] = pd.to_numeric(hiring_df["merit_score"], errors="coerce")
    hiring_df["normalized_merit_score"] = normalize_series(hiring_df["merit_score"]).round(4)
    hiring_df["low_merit_component"] = (1.0 - hiring_df["normalized_merit_score"]).round(4)
    hiring_df["connection_index"] = build_connection_index(hiring_df)
    hiring_df["discretionary_channel"] = hiring_df["discretionary_channel"].fillna("none").astype(str)
    hiring_df["discretion_factor"] = build_discretion_factor(hiring_df["discretionary_channel"]).round(4)
    hiring_df["hiring_anomaly_score"] = (
        hiring_df["hired_flag"]
        * hiring_df["low_merit_component"]
        * hiring_df["connection_index"]
        * hiring_df["discretion_factor"]
    ).round(6)

    merit_based_scores = hiring_df.loc[
        hiring_df["scenario"] == "Merit-based", "hiring_anomaly_score"
    ]
    hiring_threshold = (
        float(np.quantile(merit_based_scores, HIRING_THRESHOLD_QUANTILE))
        if not merit_based_scores.empty
        else 0.0
    )
    hiring_df["hiring_threshold"] = round(hiring_threshold, 6)
    hiring_df["hiring_anomaly_flag"] = (hiring_df["hiring_anomaly_score"] > hiring_threshold).astype(int)

    output_columns = [
        "company_id",
        "scenario",
        "employee_id",
        "candidate_id",
        "full_name",
        "merit_score",
        "connection_index",
        "discretionary_channel",
        "hiring_anomaly_score",
        "hiring_threshold",
        "hiring_anomaly_flag",
    ]
    return hiring_df[output_columns].sort_values("hiring_anomaly_score", ascending=False).reset_index(drop=True), hiring_threshold


def build_promotion_anomaly_frame(employees: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    promotion_df = employees[
        [
            "company_id",
            "scenario",
            "employee_id",
            "full_name",
            "performance_score",
            "promoted_flag",
            "close_family_relation_flag",
            "family_link_flag",
            "same_high_school_flag",
            "same_city_flag",
            "same_college_flag",
            "same_last_name_flag",
            "referral_flag",
            "connection_strength",
            "discretionary_channel",
        ]
    ].copy()

    promotion_df["performance_score"] = pd.to_numeric(promotion_df["performance_score"], errors="coerce")
    promotion_df["promoted_flag"] = pd.to_numeric(promotion_df["promoted_flag"], errors="coerce").fillna(0).astype(int)
    promotion_df["normalized_performance_score"] = normalize_series(promotion_df["performance_score"]).round(4)
    promotion_df["low_performance_component"] = (1.0 - promotion_df["normalized_performance_score"]).round(4)
    promotion_df["connection_index"] = build_connection_index(promotion_df)
    promotion_df["discretionary_channel"] = promotion_df["discretionary_channel"].fillna("none").astype(str)
    promotion_df["discretion_factor"] = build_discretion_factor(promotion_df["discretionary_channel"]).round(4)
    promotion_df["promotion_anomaly_score"] = (
        promotion_df["promoted_flag"]
        * promotion_df["low_performance_component"]
        * promotion_df["connection_index"]
        * promotion_df["discretion_factor"]
    ).round(6)

    merit_based_scores = promotion_df.loc[
        (promotion_df["scenario"] == "Merit-based") & (promotion_df["promoted_flag"] == 1),
        "promotion_anomaly_score",
    ]
    promotion_threshold = (
        float(np.quantile(merit_based_scores, PROMOTION_THRESHOLD_QUANTILE))
        if not merit_based_scores.empty
        else 0.0
    )
    promotion_df["promotion_threshold"] = round(promotion_threshold, 6)
    promotion_df["promotion_anomaly_flag"] = (
        promotion_df["promotion_anomaly_score"] > promotion_threshold
    ).astype(int)

    output_columns = [
        "company_id",
        "scenario",
        "employee_id",
        "full_name",
        "performance_score",
        "promoted_flag",
        "connection_index",
        "discretionary_channel",
        "promotion_anomaly_score",
        "promotion_threshold",
        "promotion_anomaly_flag",
    ]
    return promotion_df[output_columns].sort_values("promotion_anomaly_score", ascending=False).reset_index(drop=True), promotion_threshold


def summarize_hiring_anomalies(hiring_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        hiring_df.groupby("scenario")
        .agg(
            total_hired=("employee_id", "size"),
            anomaly_count=("hiring_anomaly_flag", "sum"),
            average_anomaly_score=("hiring_anomaly_score", "mean"),
            median_anomaly_score=("hiring_anomaly_score", "median"),
        )
        .reset_index()
    )
    summary["anomaly_rate"] = summary["anomaly_count"] / summary["total_hired"]
    return summary[
        [
            "scenario",
            "total_hired",
            "anomaly_count",
            "anomaly_rate",
            "average_anomaly_score",
            "median_anomaly_score",
        ]
    ].sort_values("scenario")


def summarize_promotion_anomalies(promotion_df: pd.DataFrame) -> pd.DataFrame:
    promoted = promotion_df[promotion_df["promoted_flag"] == 1].copy()
    summary = (
        promoted.groupby("scenario")
        .agg(
            total_promoted=("employee_id", "size"),
            anomaly_count=("promotion_anomaly_flag", "sum"),
            average_anomaly_score=("promotion_anomaly_score", "mean"),
            median_anomaly_score=("promotion_anomaly_score", "median"),
        )
        .reset_index()
    )
    summary["anomaly_rate"] = summary["anomaly_count"] / summary["total_promoted"]
    return summary[
        [
            "scenario",
            "total_promoted",
            "anomaly_count",
            "anomaly_rate",
            "average_anomaly_score",
            "median_anomaly_score",
        ]
    ].sort_values("scenario")


def plot_anomaly_rate_bar(
    summary_df: pd.DataFrame,
    scenario_column: str,
    rate_column: str,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(summary_df[scenario_column], summary_df[rate_column], color=["#6c8ebf", "#d79b00", "#c0504d"])
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Anomaly Rate")
    ax.set_title(title)
    ax.set_ylim(0, max(0.05, float(summary_df[rate_column].max()) * 1.15))
    plt.xticks(rotation=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def print_summary(
    hiring_threshold: float,
    promotion_threshold: float,
    hiring_summary: pd.DataFrame,
    promotion_summary: pd.DataFrame,
    hiring_top: pd.DataFrame,
    promotion_top: pd.DataFrame,
) -> None:
    print(f"\nHiring threshold value: {hiring_threshold:.6f}")
    print(f"Promotion threshold value: {promotion_threshold:.6f}")

    print("\nHiring anomaly rate by scenario:")
    print(hiring_summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    print("\nPromotion anomaly rate by scenario:")
    print(promotion_summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    print("\nTop 10 suspicious hires:")
    print(
        hiring_top.head(10)[
            [
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
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )

    print("\nTop 10 suspicious promotions:")
    print(
        promotion_top.head(10)[
            [
                "scenario",
                "company_id",
                "employee_id",
                "full_name",
                "performance_score",
                "promoted_flag",
                "connection_index",
                "discretionary_channel",
                "promotion_anomaly_score",
                "promotion_anomaly_flag",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )


def main() -> None:
    paths = get_paths()
    paths["output_dir"].mkdir(parents=True, exist_ok=True)

    candidates, employees = load_inputs(paths)

    hiring_df, hiring_threshold = build_hiring_anomaly_frame(candidates, employees)
    promotion_df, promotion_threshold = build_promotion_anomaly_frame(employees)

    top_hiring = hiring_df.sort_values("hiring_anomaly_score", ascending=False).head(TOP_OUTPUT_ROWS).copy()
    top_promotion = promotion_df.sort_values("promotion_anomaly_score", ascending=False).head(TOP_OUTPUT_ROWS).copy()

    hiring_summary = summarize_hiring_anomalies(hiring_df)
    promotion_summary = summarize_promotion_anomalies(promotion_df)

    hiring_df.to_csv(paths["hiring_scores"], index=False)
    promotion_df.to_csv(paths["promotion_scores"], index=False)
    top_hiring.to_csv(paths["top_hiring"], index=False)
    top_promotion.to_csv(paths["top_promotion"], index=False)
    hiring_summary.to_csv(paths["hiring_summary"], index=False)
    promotion_summary.to_csv(paths["promotion_summary"], index=False)

    plot_anomaly_rate_bar(
        hiring_summary,
        scenario_column="scenario",
        rate_column="anomaly_rate",
        title="Hiring Anomaly Rate by Scenario",
        output_path=paths["hiring_chart"],
    )
    plot_anomaly_rate_bar(
        promotion_summary,
        scenario_column="scenario",
        rate_column="anomaly_rate",
        title="Promotion Anomaly Rate by Scenario",
        output_path=paths["promotion_chart"],
    )

    print_summary(
        hiring_threshold=hiring_threshold,
        promotion_threshold=promotion_threshold,
        hiring_summary=hiring_summary,
        promotion_summary=promotion_summary,
        hiring_top=top_hiring,
        promotion_top=top_promotion,
    )

    print("\nSaved outputs:")
    print(f"  - {paths['hiring_scores']}")
    print(f"  - {paths['promotion_scores']}")
    print(f"  - {paths['top_hiring']}")
    print(f"  - {paths['top_promotion']}")
    print(f"  - {paths['hiring_summary']}")
    print(f"  - {paths['promotion_summary']}")
    print(f"  - {paths['hiring_chart']}")
    print(f"  - {paths['promotion_chart']}")


if __name__ == "__main__":
    main()
