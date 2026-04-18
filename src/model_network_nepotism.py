from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MANAGER_TOP_N = 10
DEPARTMENT_TOP_N = 10


def get_paths() -> dict[str, Path]:
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "outputs" / "network_model"
    return {
        "employees": project_root / "data" / "processed" / "employees_model.csv",
        "managers": project_root / "data" / "processed" / "manager_features.csv",
        "departments": project_root / "data" / "processed" / "department_features.csv",
        "output_dir": output_dir,
        "manager_scores": output_dir / "manager_nepotism_scores.csv",
        "department_scores": output_dir / "department_nepotism_scores.csv",
        "top_managers": output_dir / "top_risky_managers.csv",
        "top_departments": output_dir / "top_risky_departments.csv",
        "manager_chart": output_dir / "top_risky_managers.png",
        "department_chart": output_dir / "top_risky_departments.png",
    }


def load_inputs(paths: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    employees = pd.read_csv(paths["employees"])
    managers = pd.read_csv(paths["managers"])
    departments = pd.read_csv(paths["departments"])
    print(f"Loaded {len(employees)} employee rows from {paths['employees']}")
    print(f"Loaded {len(managers)} manager feature rows from {paths['managers']}")
    print(f"Loaded {len(departments)} department feature rows from {paths['departments']}")
    return employees, managers, departments


def prepare_employees(employees: pd.DataFrame) -> pd.DataFrame:
    employees = employees.copy()

    for column in [
        "family_link_flag",
        "close_family_relation_flag",
        "same_high_school_flag",
        "same_city_flag",
        "same_college_flag",
        "same_last_name_flag",
        "promoted_flag",
        "is_high_connection",
    ]:
        if column in employees.columns:
            employees[column] = pd.to_numeric(employees[column], errors="coerce").fillna(0).astype(int)

    employees["college_name"] = employees["college_name"].fillna("No College")
    employees["manager_id"] = employees["manager_id"].fillna("")
    employees["department_id"] = employees["department_id"].fillna("")

    employees["observed_connected_flag"] = (
        (employees["close_family_relation_flag"] == 1)
        | (employees["same_high_school_flag"] == 1)
        | (employees["same_city_flag"] == 1)
        | (employees["same_college_flag"] == 1)
        | (employees["same_last_name_flag"] == 1)
        | (employees["family_link_flag"] == 1)
        | (pd.to_numeric(employees["connection_strength"], errors="coerce").fillna(0.0) > 0.4)
    ).astype(int)
    return employees


def top_share(series: pd.Series, drop_values: set[str] | None = None) -> float:
    cleaned = series.dropna().astype(str)
    if drop_values:
        cleaned = cleaned[~cleaned.isin(drop_values)]
    if cleaned.empty:
        return 0.0
    return float(cleaned.value_counts(normalize=True).iloc[0])


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def concentration_component(share_flag: float, top_share_value: float, flag_weight: float = 0.5) -> float:
    return float(flag_weight * share_flag + (1.0 - flag_weight) * top_share_value)


def minmax_normalize(series: pd.Series) -> pd.Series:
    series = series.astype(float)
    minimum = float(series.min())
    maximum = float(series.max())
    if np.isclose(maximum, minimum):
        return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)
    return (series - minimum) / (maximum - minimum)


def build_manager_base(employees: pd.DataFrame, manager_features: pd.DataFrame) -> pd.DataFrame:
    reports = employees[employees["manager_id"] != ""].copy()
    manager_lookup = (
        employees[["employee_id", "full_name", "department_id", "company_id", "scenario"]]
        .drop_duplicates(subset=["employee_id", "company_id", "scenario"])
        .rename(
            columns={
                "employee_id": "manager_id",
                "full_name": "manager_full_name",
                "department_id": "manager_department_id",
            }
        )
    )

    rows: list[dict[str, object]] = []
    for (company_id, scenario, manager_id), group in reports.groupby(["company_id", "scenario", "manager_id"]):
        connected_reports = group[group["observed_connected_flag"] == 1]
        promoted_connected = connected_reports["promoted_flag"].sum()
        connected_count = len(connected_reports)

        rows.append(
            {
                "company_id": company_id,
                "scenario": scenario,
                "manager_id": manager_id,
                "headcount": len(group),
                "family_link_share_calc": group["family_link_flag"].mean(),
                "close_family_share": group["close_family_relation_flag"].mean(),
                "connected_employee_share_calc": group["observed_connected_flag"].mean(),
                "promoted_share_connected": safe_ratio(promoted_connected, connected_count),
                "surname_concentration": top_share(group["last_name"]),
                "city_concentration": top_share(group["city_name"]),
                "school_concentration": top_share(group["high_school_name"]),
                "college_concentration": top_share(group["college_name"], drop_values={"No College"}),
                "same_high_school_share": group["same_high_school_flag"].mean(),
                "same_city_share": group["same_city_flag"].mean(),
                "same_college_share": group["same_college_flag"].mean(),
                "same_last_name_share": group["same_last_name_flag"].mean(),
            }
        )

    manager_base = pd.DataFrame(rows)
    if manager_base.empty:
        return manager_base

    manager_base = manager_base.merge(
        manager_features,
        on=["company_id", "scenario", "manager_id"],
        how="left",
        suffixes=("", "_feature"),
    )
    manager_base = manager_base.merge(
        manager_lookup,
        on=["company_id", "scenario", "manager_id"],
        how="left",
    )
    manager_base["manager_full_name"] = manager_base["manager_full_name"].fillna(manager_base["manager_id"])
    manager_base["manager_department_id"] = manager_base["manager_department_id"].fillna("")
    manager_base["headcount"] = manager_base["headcount_feature"].fillna(manager_base["headcount"]) if "headcount_feature" in manager_base.columns else manager_base["headcount"]
    manager_base["family_link_share"] = manager_base["family_link_share"].fillna(manager_base["family_link_share_calc"])
    manager_base["connected_employee_share"] = manager_base["connected_employee_share"].fillna(
        manager_base["connected_employee_share_calc"]
    )
    return manager_base


def build_department_base(employees: pd.DataFrame, department_features: pd.DataFrame) -> pd.DataFrame:
    working = employees[employees["department_id"] != ""].copy()
    rows: list[dict[str, object]] = []

    for (company_id, scenario, department_id), group in working.groupby(["company_id", "scenario", "department_id"]):
        connected_group = group[group["observed_connected_flag"] == 1]
        promoted_connected = connected_group["promoted_flag"].sum()
        connected_count = len(connected_group)

        rows.append(
            {
                "company_id": company_id,
                "scenario": scenario,
                "department_id": department_id,
                "headcount": len(group),
                "family_link_share_calc": group["family_link_flag"].mean(),
                "close_family_share": group["close_family_relation_flag"].mean(),
                "connected_employee_share_calc": group["observed_connected_flag"].mean(),
                "promoted_share_connected": safe_ratio(promoted_connected, connected_count),
                "surname_concentration": top_share(group["last_name"]),
                "city_concentration": top_share(group["city_name"]),
                "school_concentration": top_share(group["high_school_name"]),
                "college_concentration": top_share(group["college_name"], drop_values={"No College"}),
                "same_high_school_share": group["same_high_school_flag"].mean(),
                "same_city_share": group["same_city_flag"].mean(),
                "same_college_share": group["same_college_flag"].mean(),
                "same_last_name_share": group["same_last_name_flag"].mean(),
            }
        )

    department_base = pd.DataFrame(rows)
    if department_base.empty:
        return department_base

    department_base = department_base.merge(
        department_features,
        on=["company_id", "scenario", "department_id"],
        how="left",
        suffixes=("", "_feature"),
    )
    department_base["headcount"] = (
        department_base["headcount_feature"].fillna(department_base["headcount"])
        if "headcount_feature" in department_base.columns
        else department_base["headcount"]
    )
    department_base["family_link_share"] = department_base["family_link_share"].fillna(
        department_base["family_link_share_calc"]
    )
    department_base["connected_employee_share"] = department_base["connected_employee_share"].fillna(
        department_base["connected_employee_share_calc"]
    )
    return department_base


def add_risk_components(df: pd.DataFrame, entity_type: str) -> pd.DataFrame:
    df = df.copy()
    df["family_concentration_component"] = (
        0.60 * df["family_link_share"] + 0.40 * df["close_family_share"]
    )
    df["school_concentration_component"] = df.apply(
        lambda row: concentration_component(row["same_high_school_share"], row["school_concentration"], 0.58),
        axis=1,
    )
    df["city_concentration_component"] = df.apply(
        lambda row: concentration_component(row["same_city_share"], row["city_concentration"], 0.42),
        axis=1,
    )
    df["college_concentration_component"] = df.apply(
        lambda row: concentration_component(row["same_college_share"], row["college_concentration"], 0.48),
        axis=1,
    )
    df["surname_concentration_component"] = df.apply(
        lambda row: concentration_component(row["same_last_name_share"], row["surname_concentration"], 0.34),
        axis=1,
    )
    df["connected_promotion_component"] = df["promoted_share_connected"]

    # Downweight extremely small units so noisy two-person groups do not dominate.
    scale = 10.0 if entity_type == "department" else 6.0
    df["size_stability_factor"] = np.clip(df["headcount"] / scale, 0.35, 1.0)

    df["raw_nepotism_risk_score"] = (
        0.28 * df["family_concentration_component"]
        + 0.20 * df["school_concentration_component"]
        + 0.16 * df["city_concentration_component"]
        + 0.12 * df["college_concentration_component"]
        + 0.08 * df["surname_concentration_component"]
        + 0.10 * df["connected_employee_share"]
        + 0.06 * df["connected_promotion_component"]
    ) * df["size_stability_factor"]

    return df


def finalize_scores(df: pd.DataFrame, score_column: str) -> pd.DataFrame:
    df = df.copy()
    df[score_column] = minmax_normalize(df["raw_nepotism_risk_score"]).round(4)
    df["raw_nepotism_risk_score"] = df["raw_nepotism_risk_score"].round(4)
    for column in [
        "family_concentration_component",
        "school_concentration_component",
        "city_concentration_component",
        "college_concentration_component",
        "surname_concentration_component",
        "connected_employee_share",
        "connected_promotion_component",
        "family_link_share",
        "close_family_share",
        "promoted_share_connected",
        "surname_concentration",
        "city_concentration",
        "school_concentration",
        "college_concentration",
        "same_high_school_share",
        "same_city_share",
        "same_college_share",
        "same_last_name_share",
        "size_stability_factor",
    ]:
        if column in df.columns:
            df[column] = df[column].astype(float).round(4)
    return df.sort_values(score_column, ascending=False).reset_index(drop=True)


def plot_top_entities(
    df: pd.DataFrame,
    label_column: str,
    score_column: str,
    title: str,
    output_path: Path,
) -> None:
    top_df = df.head(10).copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_df[label_column][::-1], top_df[score_column][::-1], color="#c75b39")
    ax.set_xlabel("Risk Score")
    ax.set_ylabel("")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def print_summary(manager_scores: pd.DataFrame, department_scores: pd.DataFrame) -> None:
    print("\nAverage manager risk score by scenario:")
    manager_avg = manager_scores.groupby("scenario")["manager_nepotism_risk_score"].mean().sort_values(ascending=False)
    for scenario, value in manager_avg.items():
        print(f"  - {scenario}: {value:.4f}")

    print("\nAverage department risk score by scenario:")
    department_avg = (
        department_scores.groupby("scenario")["department_nepotism_risk_score"].mean().sort_values(ascending=False)
    )
    for scenario, value in department_avg.items():
        print(f"  - {scenario}: {value:.4f}")

    print("\nTop 10 risky managers:")
    manager_view = manager_scores.head(10)[
        [
            "scenario",
            "company_id",
            "manager_id",
            "manager_full_name",
            "headcount",
            "manager_nepotism_risk_score",
        ]
    ]
    print(manager_view.to_string(index=False))

    print("\nTop 10 risky departments:")
    department_view = department_scores.head(10)[
        [
            "scenario",
            "company_id",
            "department_id",
            "headcount",
            "department_nepotism_risk_score",
        ]
    ]
    print(department_view.to_string(index=False))


def main() -> None:
    paths = get_paths()
    paths["output_dir"].mkdir(parents=True, exist_ok=True)

    employees, manager_features, department_features = load_inputs(paths)
    employees = prepare_employees(employees)

    manager_scores = build_manager_base(employees, manager_features)
    manager_scores = add_risk_components(manager_scores, entity_type="manager")
    manager_scores = finalize_scores(manager_scores, "manager_nepotism_risk_score")

    department_scores = build_department_base(employees, department_features)
    department_scores = add_risk_components(department_scores, entity_type="department")
    department_scores = finalize_scores(department_scores, "department_nepotism_risk_score")

    top_managers = manager_scores.head(MANAGER_TOP_N).copy()
    top_departments = department_scores.head(DEPARTMENT_TOP_N).copy()

    manager_scores.to_csv(paths["manager_scores"], index=False)
    department_scores.to_csv(paths["department_scores"], index=False)
    top_managers.to_csv(paths["top_managers"], index=False)
    top_departments.to_csv(paths["top_departments"], index=False)

    plot_top_entities(
        top_managers.assign(label=lambda df: df["manager_full_name"] + " (" + df["manager_id"] + ")"),
        label_column="label",
        score_column="manager_nepotism_risk_score",
        title="Top Risky Managers",
        output_path=paths["manager_chart"],
    )
    plot_top_entities(
        top_departments.assign(label=lambda df: df["company_id"] + " / " + df["department_id"]),
        label_column="label",
        score_column="department_nepotism_risk_score",
        title="Top Risky Departments",
        output_path=paths["department_chart"],
    )

    print_summary(manager_scores, department_scores)
    print("\nSaved outputs:")
    print(f"  - {paths['manager_scores']}")
    print(f"  - {paths['department_scores']}")
    print(f"  - {paths['top_managers']}")
    print(f"  - {paths['top_departments']}")
    print(f"  - {paths['manager_chart']}")
    print(f"  - {paths['department_chart']}")


if __name__ == "__main__":
    main()
