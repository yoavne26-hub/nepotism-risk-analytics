from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from src.nonnegative_logistic import NonNegativeLogisticRegression
except ModuleNotFoundError:  # pragma: no cover - script execution fallback
    from nonnegative_logistic import NonNegativeLogisticRegression


CHANNEL_FEATURES = [
    "has_discretionary_channel",
    "discretionary_family_flag",
    "discretionary_referral_flag",
    "discretionary_manager_endorsement_flag",
    "discretionary_alumni_network_flag",
    "discretionary_legacy_referral_flag",
    "discretionary_internal_source_flag",
]

EXPLANATORY_FEATURES = [
    "performance_score",
    "tenure_months",
    "promotion_headroom",
    "years_experience",
    "close_family_relation_flag",
    "same_high_school_flag",
    "same_city_flag",
    "same_college_flag",
    "same_last_name_flag",
    "family_link_flag",
    "connection_strength",
    *CHANNEL_FEATURES,
]

COMPACT_FEATURES = [
    "performance_score",
    "tenure_months",
    "promotion_headroom",
    "years_experience",
    "close_family_relation_flag",
    "same_high_school_flag",
    "same_city_flag",
    "same_college_flag",
    "same_last_name_flag",
    "family_link_flag",
    "connection_strength",
    *CHANNEL_FEATURES,
]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    display_name: str
    layer: str
    feature_list: list[str]
    interaction_bases: list[str]
    class_weight: str | None = None
    calibrated: bool = False
    supports_coefficients: bool = True


@dataclass
class FeatureSchema:
    feature_list: list[str]
    interaction_bases: list[str]
    medians: pd.Series
    categories: list[str]
    feature_columns: list[str]


MODEL_SPECS = [
    ModelSpec(
        name="balanced_explanatory",
        display_name="Balanced Explanatory Logistic",
        layer="explanatory",
        feature_list=EXPLANATORY_FEATURES,
        interaction_bases=[],
        class_weight="balanced",
    ),
    ModelSpec(
        name="unweighted_explanatory",
        display_name="Unweighted Explanatory Logistic",
        layer="explanatory",
        feature_list=EXPLANATORY_FEATURES,
        interaction_bases=[],
    ),
    ModelSpec(
        name="calibrated_explanatory",
        display_name="Calibrated Explanatory Logistic",
        layer="explanatory",
        feature_list=EXPLANATORY_FEATURES,
        interaction_bases=[],
        calibrated=True,
        supports_coefficients=False,
    ),
    ModelSpec(
        name="balanced_compact",
        display_name="Balanced Compact Logistic",
        layer="compact",
        feature_list=COMPACT_FEATURES,
        interaction_bases=[],
        class_weight="balanced",
    ),
    ModelSpec(
        name="unweighted_compact",
        display_name="Unweighted Compact Logistic",
        layer="compact",
        feature_list=COMPACT_FEATURES,
        interaction_bases=[],
    ),
    ModelSpec(
        name="calibrated_compact",
        display_name="Calibrated Compact Logistic",
        layer="compact",
        feature_list=COMPACT_FEATURES,
        interaction_bases=[],
        calibrated=True,
        supports_coefficients=False,
    ),
]


def get_paths() -> dict[str, Path]:
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "outputs" / "promotion_model"
    return {
        "input": project_root / "data" / "processed" / "employees_model.csv",
        "output_dir": output_dir,
        "metrics": output_dir / "promotion_metrics.csv",
        "coefficients": output_dir / "promotion_coefficients.csv",
        "predictions": output_dir / "employee_predictions.csv",
        "roc_curve": output_dir / "roc_curve.png",
        "pr_curve": output_dir / "pr_curve.png",
    }


def load_data(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} employee rows from {input_path}")
    return df


def ensure_model_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    numeric_defaults = {
        "performance_score": np.nan,
        "tenure_months": 0.0,
        "salary": np.nan,
        "role_level": np.nan,
        "years_experience": 0.0,
        "merit_score": np.nan,
        "close_family_relation_flag": 0,
        "same_high_school_flag": 0,
        "same_city_flag": 0,
        "same_college_flag": 0,
        "same_last_name_flag": 0,
        "family_link_flag": 0,
        "connection_strength": 0.0,
    }
    for column, default_value in numeric_defaults.items():
        if column not in df.columns:
            df[column] = default_value

    if "discretionary_channel" not in df.columns:
        df["discretionary_channel"] = "none"

    for column in numeric_defaults:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["promoted_flag"] = pd.to_numeric(df["promoted_flag"], errors="raise").astype(int)
    df["discretionary_channel"] = df["discretionary_channel"].fillna("missing").astype(str)
    df["promotion_headroom"] = np.maximum(1.0, 6.0 - df["role_level"].fillna(5.0))
    df["has_discretionary_channel"] = (df["discretionary_channel"] != "none").astype(int)
    df["discretionary_family_flag"] = (df["discretionary_channel"] == "family").astype(int)
    df["discretionary_referral_flag"] = (df["discretionary_channel"] == "referral").astype(int)
    df["discretionary_manager_endorsement_flag"] = (
        df["discretionary_channel"] == "manager_endorsement"
    ).astype(int)
    df["discretionary_alumni_network_flag"] = (
        df["discretionary_channel"] == "alumni_network"
    ).astype(int)
    df["discretionary_legacy_referral_flag"] = (
        df["discretionary_channel"] == "legacy_referral"
    ).astype(int)
    df["discretionary_internal_source_flag"] = (
        df["discretionary_channel"] == "internal_source"
    ).astype(int)
    return df


def fit_feature_schema(
    train_df: pd.DataFrame,
    feature_list: list[str],
    interaction_bases: list[str],
) -> FeatureSchema:
    numeric_columns = [column for column in feature_list if column != "discretionary_channel"]
    medians = train_df[numeric_columns].median(numeric_only=True)
    categories: list[str] = []
    if "discretionary_channel" in feature_list:
        categories = sorted(train_df["discretionary_channel"].fillna("missing").astype(str).unique())

    schema = FeatureSchema(
        feature_list=feature_list,
        interaction_bases=interaction_bases,
        medians=medians,
        categories=categories,
        feature_columns=[],
    )
    schema.feature_columns = transform_features(train_df, schema).columns.tolist()
    return schema


def transform_features(df: pd.DataFrame, schema: FeatureSchema) -> pd.DataFrame:
    numeric_columns = [column for column in schema.feature_list if column != "discretionary_channel"]
    feature_df = df[numeric_columns].copy().fillna(schema.medians).reset_index(drop=True)

    dummy_columns: list[str] = []
    if "discretionary_channel" in schema.feature_list:
        channels = pd.Categorical(
            df["discretionary_channel"].fillna("missing").astype(str),
            categories=schema.categories,
        )
        dummies = pd.get_dummies(channels, prefix="discretionary_channel", dtype=int)
        dummy_columns = dummies.columns.tolist()
        feature_df = pd.concat([feature_df, dummies.reset_index(drop=True)], axis=1)

    for base_column in schema.interaction_bases:
        if base_column not in feature_df.columns:
            continue
        for dummy_column in dummy_columns:
            feature_df[f"{base_column}_x_{dummy_column}"] = (
                feature_df[base_column] * feature_df[dummy_column]
            )

    if schema.feature_columns:
        for column in schema.feature_columns:
            if column not in feature_df.columns:
                feature_df[column] = 0.0
        feature_df = feature_df[schema.feature_columns]

    return feature_df


def fit_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    class_weight: str | None,
) -> NonNegativeLogisticRegression:
    model = NonNegativeLogisticRegression(
        max_iter=5000,
        class_weight=class_weight,
        random_state=42,
        min_coef=1e-6,
    )
    model.fit(X_train, y_train)
    return model


def fit_model(spec: ModelSpec, X_train: pd.DataFrame, y_train: pd.Series):
    if spec.calibrated:
        positives = int(y_train.sum())
        cv_folds = max(2, min(3, positives))
        base_model = NonNegativeLogisticRegression(
            max_iter=5000,
            class_weight=spec.class_weight,
            random_state=42,
            min_coef=1e-6,
        )
        calibrated_model = CalibratedClassifierCV(estimator=base_model, method="sigmoid", cv=cv_folds)
        calibrated_model.fit(X_train, y_train)
        return calibrated_model

    return fit_logistic_regression(X_train, y_train, spec.class_weight)


def evaluate_model(
    spec: ModelSpec,
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_all: pd.DataFrame,
) -> tuple[dict[str, float | str | bool], np.ndarray, np.ndarray]:
    test_probabilities = model.predict_proba(X_test)[:, 1]
    all_probabilities = model.predict_proba(X_all)[:, 1]
    test_predictions = (test_probabilities >= 0.5).astype(int)

    metrics = {
        "model": spec.name,
        "display_name": spec.display_name,
        "layer": spec.layer,
        "class_weight": spec.class_weight or "none",
        "calibrated": spec.calibrated,
        "roc_auc": roc_auc_score(y_test, test_probabilities),
        "precision": precision_score(y_test, test_predictions, zero_division=0),
        "recall": recall_score(y_test, test_predictions, zero_division=0),
        "f1": f1_score(y_test, test_predictions, zero_division=0),
        "brier_score": brier_score_loss(y_test, test_probabilities),
        "avg_predicted_probability_test": float(np.mean(test_probabilities)),
        "avg_predicted_probability_all": float(np.mean(all_probabilities)),
    }
    return metrics, test_probabilities, all_probabilities


def build_coefficient_table(
    model_name: str,
    layer: str,
    model,
    feature_names: list[str],
    scenario: str = "all",
) -> pd.DataFrame:
    coefficients = pd.DataFrame(
        {
            "model": model_name,
            "layer": layer,
            "scenario": scenario,
            "feature": feature_names,
            "coefficient": model.coef_.ravel(),
        }
    )
    coefficients["abs_coefficient"] = coefficients["coefficient"].abs()
    coefficients["odds_ratio"] = np.exp(coefficients["coefficient"])
    return coefficients.sort_values(["model", "abs_coefficient"], ascending=[True, False])


def fit_scenario_specific_explanatory_models(df: pd.DataFrame) -> tuple[list[pd.DataFrame], list[str]]:
    coefficient_tables: list[pd.DataFrame] = []
    notes: list[str] = []

    for scenario_name, scenario_df in df.groupby("scenario"):
        y = scenario_df["promoted_flag"].astype(int)
        positives = int(y.sum())
        if len(scenario_df) < 45 or positives < 4 or y.nunique() < 2:
            notes.append(
                f"Skipped scenario-specific explanatory promotion model for {scenario_name} "
                f"(rows={len(scenario_df)}, promotions={positives})."
            )
            continue

        schema = fit_feature_schema(
            scenario_df,
            EXPLANATORY_FEATURES,
            [],
        )
        X = transform_features(scenario_df, schema)
        model = fit_logistic_regression(X, y, class_weight="balanced")
        coefficient_tables.append(
            build_coefficient_table(
                model_name=f"balanced_explanatory__{scenario_name}",
                layer="explanatory_scenario",
                model=model,
                feature_names=schema.feature_columns,
                scenario=scenario_name,
            )
        )

    return coefficient_tables, notes


def plot_curves(
    y_test: pd.Series,
    probability_map: dict[str, tuple[str, np.ndarray]],
    roc_path: Path,
    pr_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for _, (label, probabilities) in probability_map.items():
        fpr, tpr, _ = roc_curve(y_test, probabilities)
        ax.plot(fpr, tpr, label=label)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Promotion Model ROC Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(roc_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    for _, (label, probabilities) in probability_map.items():
        precision, recall, _ = precision_recall_curve(y_test, probabilities)
        ax.plot(recall, precision, label=label)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Promotion Model Precision-Recall Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(pr_path, dpi=150)
    plt.close(fig)


def print_coefficient_table(coefficients_df: pd.DataFrame, model_name: str, header: str) -> None:
    subset = coefficients_df[coefficients_df["model"] == model_name][
        ["feature", "coefficient", "odds_ratio"]
    ].sort_values("coefficient", ascending=False)
    print(f"\n{header}:")
    print(subset.to_string(index=False, float_format=lambda value: f"{value:0.4f}"))


def print_probability_summary(observed_rate: float, metrics_df: pd.DataFrame) -> None:
    print(f"\nObserved promotion rate: {observed_rate:.3%}")
    print("\nAverage predicted probability by model:")
    for row in metrics_df.sort_values(["layer", "model"]).itertuples(index=False):
        print(
            f"  - {row.display_name}: test={row.avg_predicted_probability_test:.3%}, "
            f"all={row.avg_predicted_probability_all:.3%}"
        )


def print_model_summary(metrics_df: pd.DataFrame, scenario_notes: list[str]) -> None:
    scoring_best = metrics_df.sort_values("brier_score").iloc[0]
    print("\nSummary:")
    print(
        "  - Interpretation: Balanced Explanatory Logistic is the main interpretation model "
        "because it keeps promotion merit dimensions separate from the connection variables."
    )
    print(
        f"  - Realistic probability scoring: {scoring_best['display_name']} is best on this run "
        "because it has the lowest Brier score."
    )
    print(
        "  - Compact layer: the compact models provide a shorter positive-feature specification "
        "without using the visual merit score as a predictor."
    )
    if scenario_notes:
        print("\nScenario-specific explanatory models:")
        for note in scenario_notes:
            print(f"  - {note}")
    else:
        print("\nScenario-specific explanatory models were fit and saved in the coefficients output.")


def main() -> None:
    paths = get_paths()
    paths["output_dir"].mkdir(parents=True, exist_ok=True)

    df = ensure_model_features(load_data(paths["input"]))
    observed_promotion_rate = float(df["promoted_flag"].mean())

    train_df, test_df = train_test_split(
        df,
        test_size=0.30,
        random_state=42,
        stratify=df["promoted_flag"],
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    all_df = df.reset_index(drop=True)
    y_train = train_df["promoted_flag"]
    y_test = test_df["promoted_flag"]

    metrics_rows = []
    coefficient_tables = []
    probability_map: dict[str, tuple[str, np.ndarray]] = {}

    test_index_labels = set(test_df["employee_id"].astype(str))
    predictions_df = pd.DataFrame(
        {
            "employee_id": all_df.get("employee_id"),
            "candidate_id": all_df.get("candidate_id"),
            "company_id": all_df.get("company_id"),
            "scenario": all_df.get("scenario"),
            "promoted_flag": all_df["promoted_flag"],
            "split": np.where(
                all_df["employee_id"].astype(str).isin(test_index_labels),
                "test",
                "train",
            ),
        }
    )

    for spec in MODEL_SPECS:
        schema = fit_feature_schema(train_df, spec.feature_list, spec.interaction_bases)
        X_train = transform_features(train_df, schema)
        X_test = transform_features(test_df, schema)
        X_all = transform_features(all_df, schema)

        model = fit_model(spec, X_train, y_train)
        metrics_row, test_probabilities, all_probabilities = evaluate_model(
            spec, model, X_test, y_test, X_all
        )
        metrics_rows.append(metrics_row)
        probability_map[spec.name] = (spec.display_name, test_probabilities)
        predictions_df[f"{spec.name}_prob"] = all_probabilities
        predictions_df[f"{spec.name}_pred"] = (all_probabilities >= 0.5).astype(int)

        if spec.supports_coefficients:
            coefficient_tables.append(
                build_coefficient_table(
                    model_name=spec.name,
                    layer=spec.layer,
                    model=model,
                    feature_names=schema.feature_columns,
                )
            )

    scenario_coefficient_tables, scenario_notes = fit_scenario_specific_explanatory_models(df)
    coefficient_tables.extend(scenario_coefficient_tables)

    metrics_df = pd.DataFrame(metrics_rows).sort_values(["layer", "model"]).reset_index(drop=True)
    coefficients_df = pd.concat(coefficient_tables, ignore_index=True)

    metrics_df.to_csv(paths["metrics"], index=False)
    coefficients_df.to_csv(paths["coefficients"], index=False)
    predictions_df.to_csv(paths["predictions"], index=False)
    plot_curves(y_test, probability_map, paths["roc_curve"], paths["pr_curve"])

    print_coefficient_table(
        coefficients_df,
        model_name="balanced_explanatory",
        header="Balanced explanatory-model coefficients",
    )
    print_coefficient_table(
        coefficients_df,
        model_name="balanced_compact",
        header="Balanced compact-model coefficients",
    )
    print_probability_summary(observed_promotion_rate, metrics_df)
    print("\nSaved outputs:")
    print(f"  - {paths['metrics']}")
    print(f"  - {paths['coefficients']}")
    print(f"  - {paths['predictions']}")
    print(f"  - {paths['roc_curve']}")
    print(f"  - {paths['pr_curve']}")
    print_model_summary(metrics_df, scenario_notes)


if __name__ == "__main__":
    main()
