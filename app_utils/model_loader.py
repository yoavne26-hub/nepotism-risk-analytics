from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "predictor_models"
ARTIFACT_VERSION = "predictor-bundles-v1"


@dataclass
class LiveModel:
    spec_name: str
    display_name: str
    layer: str
    calibrated: bool
    class_weight: str | None
    model: Any
    schema: Any
    training_rows: int
    positive_rate: float
    brier_score: float | None
    roc_auc: float | None
    source_note: str
    artifact_status: str
    artifact_path: str | None


@dataclass
class TaskBundle:
    task: str
    target_column: str
    source_data_path: str
    default_model_name: str
    models: dict[str, LiveModel]
    reference_data: dict[str, Any]
    artifact_status: str
    artifact_path: str | None


MODULE_PATHS = {
    "hiring": "src.model_hiring",
    "promotion": "src.model_promotion",
}

TARGET_COLUMNS = {
    "hiring": "hired_flag",
    "promotion": "promoted_flag",
}

DEFAULT_MODEL_FALLBACKS = {
    "hiring": "unweighted_compact",
    "promotion": "unweighted_compact",
}

REFERENCE_NUMERIC_COLUMNS = {
    "hiring": [
        "high_school_gpa",
        "college_gpa",
        "test_score",
        "interview_score",
        "years_experience",
        "merit_score",
        "connection_strength",
    ],
    "promotion": [
        "performance_score",
        "tenure_months",
        "salary",
        "role_level",
        "years_experience",
        "merit_score",
        "connection_strength",
    ],
}


def _get_module(task: str):
    if task not in MODULE_PATHS:
        raise ValueError(f"Unsupported task: {task}")
    return import_module(MODULE_PATHS[task])


def _artifact_path(task: str) -> Path:
    return ARTIFACT_ROOT / f"{task}_bundle.joblib"


def _path_state(path: Path) -> dict[str, int | bool | str]:
    if not path.exists():
        return {"path": str(path), "exists": False}

    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "mtime_ns": int(stat.st_mtime_ns),
        "size": int(stat.st_size),
    }


def _build_signature(task: str, module, paths: dict[str, Path]) -> dict[str, Any]:
    return {
        "artifact_version": ARTIFACT_VERSION,
        "task": task,
        "module_name": module.__name__,
        "module_state": _path_state(Path(module.__file__).resolve()),
        "input_state": _path_state(paths["input"]),
        "metrics_state": _path_state(paths["metrics"]),
        "model_specs": [spec.name for spec in module.MODEL_SPECS],
    }


def _read_metrics(metrics_path: Path) -> pd.DataFrame:
    if not metrics_path.exists():
        return pd.DataFrame()
    return pd.read_csv(metrics_path)


def _build_numeric_ranges(df: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for column in columns:
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        summary[column] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "median": float(series.median()),
        }
    return summary


def _build_reference_data(task: str, df: pd.DataFrame) -> dict[str, Any]:
    reference = {
        "discretionary_channels": sorted(df["discretionary_channel"].fillna("missing").astype(str).unique().tolist()),
        "numeric_ranges": _build_numeric_ranges(df, REFERENCE_NUMERIC_COLUMNS[task]),
    }
    if task == "hiring":
        reference["education_levels"] = sorted(df["education_level"].fillna("missing").astype(str).unique().tolist())
    else:
        reference["role_levels"] = sorted(
            pd.to_numeric(df["role_level"], errors="coerce").dropna().astype(int).unique().tolist()
        )
    return reference


def _resolve_default_model_name(task: str, available_models: set[str], metrics_df: pd.DataFrame) -> str:
    if not metrics_df.empty:
        best_model_name = metrics_df.sort_values("brier_score", ascending=True)["model"].astype(str).iloc[0]
        if best_model_name in available_models:
            return best_model_name

    fallback = DEFAULT_MODEL_FALLBACKS[task]
    if fallback in available_models:
        return fallback
    return sorted(available_models)[0]


def _source_note(status: str, artifact_path: Path | None, source_data_path: str) -> str:
    if status == "Persisted artifact" and artifact_path is not None:
        try:
            rel_path = artifact_path.relative_to(PROJECT_ROOT)
        except ValueError:
            rel_path = artifact_path
        return f"Loaded from persisted artifact `{rel_path.as_posix()}` built from {source_data_path}."
    return f"Fitted from {source_data_path} and cached in memory for this app session."


def _build_task_bundle(task: str, module, paths: dict[str, Path], artifact_path: Path | None, artifact_status: str) -> TaskBundle:
    target_column = TARGET_COLUMNS[task]
    source_data_path = str(paths["input"])
    metrics_df = _read_metrics(paths["metrics"])

    df = module.ensure_model_features(pd.read_csv(paths["input"]))
    y = df[target_column].astype(int)

    metrics_lookup: dict[str, dict[str, Any]] = {}
    if not metrics_df.empty:
        metrics_lookup = {
            str(row["model"]): row
            for _, row in metrics_df.set_index("model", drop=False).iterrows()
        }

    trained_models: dict[str, LiveModel] = {}
    for spec in module.MODEL_SPECS:
        schema = module.fit_feature_schema(df, spec.feature_list, spec.interaction_bases)
        X = module.transform_features(df, schema)
        trained = module.fit_model(spec, X, y)
        metric_row = metrics_lookup.get(spec.name, {})
        trained_models[spec.name] = LiveModel(
            spec_name=spec.name,
            display_name=spec.display_name,
            layer=spec.layer,
            calibrated=bool(spec.calibrated),
            class_weight=spec.class_weight,
            model=trained,
            schema=schema,
            training_rows=len(df),
            positive_rate=float(y.mean()),
            brier_score=float(metric_row["brier_score"]) if "brier_score" in metric_row else None,
            roc_auc=float(metric_row["roc_auc"]) if "roc_auc" in metric_row else None,
            source_note=_source_note(artifact_status, artifact_path, source_data_path),
            artifact_status=artifact_status,
            artifact_path=str(artifact_path) if artifact_path is not None else None,
        )

    return TaskBundle(
        task=task,
        target_column=target_column,
        source_data_path=source_data_path,
        default_model_name=_resolve_default_model_name(task, set(trained_models.keys()), metrics_df),
        models=trained_models,
        reference_data=_build_reference_data(task, df),
        artifact_status=artifact_status,
        artifact_path=str(artifact_path) if artifact_path is not None else None,
    )


def _load_valid_artifact(task: str, module, paths: dict[str, Path]) -> TaskBundle | None:
    artifact_path = _artifact_path(task)
    if not artifact_path.exists():
        return None

    try:
        payload = joblib.load(artifact_path)
    except Exception:
        return None

    expected_signature = _build_signature(task, module, paths)
    if payload.get("signature") != expected_signature:
        return None

    bundle = payload.get("bundle")
    if not isinstance(bundle, TaskBundle):
        return None
    return bundle


def _persist_bundle(task: str, bundle: TaskBundle, signature: dict[str, Any]) -> None:
    artifact_path = _artifact_path(task)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"signature": signature, "bundle": bundle}, artifact_path)


@lru_cache(maxsize=4)
def load_task_bundle(task: str) -> TaskBundle:
    module = _get_module(task)
    paths = module.get_paths()

    persisted_bundle = _load_valid_artifact(task, module, paths)
    if persisted_bundle is not None:
        return persisted_bundle

    artifact_path = _artifact_path(task)
    built_bundle = _build_task_bundle(
        task=task,
        module=module,
        paths=paths,
        artifact_path=artifact_path,
        artifact_status="Persisted artifact",
    )
    signature = _build_signature(task, module, paths)

    try:
        _persist_bundle(task, built_bundle, signature)
        return built_bundle
    except Exception:
        return _build_task_bundle(
            task=task,
            module=module,
            paths=paths,
            artifact_path=None,
            artifact_status="Live fit fallback",
        )


def get_task_bundle(task: str) -> TaskBundle:
    return load_task_bundle(task)


def get_model_choices(bundle: TaskBundle) -> list[dict[str, str]]:
    model_choices = []
    for model_name, live_model in bundle.models.items():
        suffix = " (Recommended)" if model_name == bundle.default_model_name else ""
        label = f"{live_model.display_name}{suffix}"
        model_choices.append({"name": model_name, "label": label})
    return sorted(model_choices, key=lambda item: (item["name"] != bundle.default_model_name, item["label"]))
