"""
FastAPI ML endpoints.
All ML status, metrics, predictions, and training triggers.
"""

import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from backend.services.ml_service import ml_service

router = APIRouter(prefix="/api/ml", tags=["Machine Learning"])

BASE_DIR = Path(__file__).resolve().parent.parent.parent


# ─── Request models ──────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    train_age_years: float = 15.0
    average_speed_kmph: float = 80.0
    distance_travelled_km: float = 100000
    ambient_temperature_c: float = 30.0
    humidity_percent: float = 65.0
    rainfall_mm: float = 5.0
    wheel_wear_percent: float = 40.0
    track_vibration_level: float = 3.5
    rail_wear_mm: float = 9.0
    bearing_temperature_c: float = 76.0
    axle_temperature_c: float = 64.0
    brake_pad_wear_percent: float = 42.0
    brake_pressure_psi: float = 97.0
    battery_voltage: float = 24.4
    last_maintenance_days: float = 180.0
    sensor_health_index: float = 65.0
    inspection_score: float = 65.0
    delay_minutes: float = 12.0
    region: str = "Northern Railway"
    season: str = "Summer"
    train_type: str = "Passenger"
    # Optional DS2 extras (if available)
    wind_speed_kmph: float | None = None
    track_temperature_c: float | None = None
    traction_motor_temp_c: float | None = None
    power_consumption_kw: float | None = None
    load_factor_percent: float | None = None
    daily_trips: int | None = None
    ballast_condition: str | None = None
    signal_system_status: str | None = None
    track_curvature_degree: float | None = None


# ─── Endpoints ───────────────────────────────────────────────────────────────

@router.get("/status")
def get_ml_status() -> Dict[str, Any]:
    """Return per-model readiness status from trained artifacts."""
    ml_service.load_models()
    return ml_service.get_status()


@router.get("/metrics")
def get_model_metrics() -> Dict[str, Any]:
    """Return actual evaluation metrics from last training run."""
    ml_service.load_models()
    if not ml_service.metrics:
        return {
            "status": "NOT_TRAINED",
            "message": "No metrics available. Run: python train_all_models.py",
        }
    return ml_service.metrics


@router.get("/dataset-report")
def get_dataset_report() -> Dict[str, Any]:
    """Return the dataset inspection report generated during training."""
    report_path = BASE_DIR / "ml" / "artifacts" / "reports" / "dataset_report.json"
    if not report_path.exists():
        return {
            "status": "NOT_AVAILABLE",
            "message": "Dataset report not generated yet. Run training first.",
        }
    import json
    with open(report_path, "r", encoding="utf-8") as f:
        return json.load(f)


@router.get("/features")
def get_feature_schemas() -> Dict[str, Any]:
    """Return feature schemas for each trained model."""
    schemas = {}
    for model_key in ["m1_priority", "m3_failure", "m3_severity"]:
        schema_path = BASE_DIR / "ml" / "artifacts" / f"feature_schema_{model_key}.json"
        if schema_path.exists():
            import json
            with open(schema_path, "r", encoding="utf-8") as f:
                schemas[model_key] = json.load(f)
        else:
            schemas[model_key] = {"status": "NOT_TRAINED"}
    return schemas


@router.get("/explanations")
def get_shap_explanations() -> Dict[str, Any]:
    """Return saved SHAP summary reports."""
    explanations = {}
    for model_key in ["m1_priority", "m3_failure"]:
        shap_path = BASE_DIR / "ml" / "artifacts" / "reports" / f"shap_{model_key}.json"
        if shap_path.exists():
            import json
            with open(shap_path, "r", encoding="utf-8") as f:
                explanations[model_key] = json.load(f)
        else:
            explanations[model_key] = {"status": "NOT_TRAINED"}
    return explanations


@router.post("/predict")
def predict(req: PredictRequest) -> Dict[str, Any]:
    """
    Run M1 + M3 + M3-Severity inference.
    All predictions come from trained models — never hardcoded.
    """
    if not ml_service.is_trained:
        return {
            "status": "MODEL_NOT_TRAINED",
            "message": "No trained models available. Run: python train_all_models.py",
        }
    input_row = req.model_dump(exclude_none=True)
    return ml_service.predict(input_row)


@router.post("/train")
def trigger_training(background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """
    Trigger the ML training pipeline in the background.
    Uses real uploaded datasets from ml/data/raw/.
    """
    csv_files = []
    seen = set()
    for raw_dir in [BASE_DIR / "data" / "raw", BASE_DIR / "ml" / "data" / "raw"]:
        if raw_dir.exists():
            for p in raw_dir.glob("*.csv"):
                if p.name not in seen:
                    seen.add(p.name)
                    csv_files.append(p)

    if not csv_files:
        raise HTTPException(
            status_code=400,
            detail=(
                "No CSV datasets found in data/raw/ or ml/data/raw/. "
                "Please upload your railway maintenance datasets first."
            ),
        )

    def _run():
        script = BASE_DIR / "train_all_models.py"
        subprocess.run([sys.executable, str(script)], check=False)
        ml_service.load_models()

    background_tasks.add_task(_run)
    return {
        "status": "TRAINING_STARTED",
        "datasets": [f.name for f in csv_files],
        "message": "Training pipeline launched. Check /api/ml/status for progress.",
    }
