"""
RailwayMLService: Unified model loader and inference engine for RailBlock AI.
Connects trained ML artifacts with FastAPI and the AI Decision Engine.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import joblib

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_ARTIFACTS = _PROJECT_ROOT / "ml" / "artifacts"


class RailwayMLService:
    def __init__(self, artifacts_dir: Optional[Path] = None):
        self.artifacts_dir = Path(artifacts_dir or _DEFAULT_ARTIFACTS)
        self.models_dir = self.artifacts_dir / "models"

        # Model handles
        self.m1 = None          # Maintenance Priority (XGBoost Regressor)
        self.m3 = None          # Failure Risk (XGBoost Classifier)
        self.m3_sev = None      # Failure Severity (XGBoost Multiclass)

        # Feature and category metadata
        self.m1_features: List[str] = []
        self.m3_features: List[str] = []
        self.m3_sev_features: List[str] = []
        self.m1_cat_cats: Dict[str, List[str]] = {}
        self.m3_cat_cats: Dict[str, List[str]] = {}
        self.m3_sev_cat_cats: Dict[str, List[str]] = {}
        self.m3_sev_label_classes: List[str] = []
        self.m1_std_err: float = 0.0

        self.metrics: Dict[str, Any] = {}
        self.status: Dict[str, Any] = {}

        self.m1_ready = False
        self.m3_ready = False
        self.m3_sev_ready = False

        self.load_models()

    def _find_model_file(self, *filenames: str) -> Optional[Path]:
        for fn in filenames:
            p1 = self.models_dir / fn
            if p1.exists():
                return p1
            p2 = self.artifacts_dir / fn
            if p2.exists():
                return p2
        return None

    def load_models(self) -> bool:
        """Loads all trained model artifacts from the artifacts directory."""
        any_loaded = False

        # 1. M1: Priority
        m1_path = self._find_model_file("maintenance_priority_model.joblib", "m1_priority.joblib")
        if m1_path:
            try:
                data = joblib.load(m1_path)
                self.m1 = data["model"]
                self.m1_features = data.get("features", [])
                self.m1_cat_cats = data.get("cat_categories", {})
                self.m1_std_err = data.get("std_err", 0.0)
                self.m1_ready = True
                any_loaded = True
            except Exception as e:
                print(f"  [ML Service] Error loading M1: {e}")

        # 2. M3: Failure Risk
        m3_path = self._find_model_file("failure_risk_model.joblib", "m3_failure.joblib")
        if m3_path:
            try:
                data = joblib.load(m3_path)
                self.m3 = data["model"]
                self.m3_features = data.get("features", [])
                self.m3_cat_cats = data.get("cat_categories", {})
                self.m3_ready = True
                any_loaded = True
            except Exception as e:
                print(f"  [ML Service] Error loading M3: {e}")

        # 3. M3: Severity
        sev_path = self._find_model_file("failure_severity_model.joblib", "m3_severity.joblib")
        if sev_path:
            try:
                data = joblib.load(sev_path)
                self.m3_sev = data["model"]
                self.m3_sev_features = data.get("features", [])
                self.m3_sev_cat_cats = data.get("cat_categories", {})
                self.m3_sev_label_classes = data.get("label_encoder_classes", [])
                self.m3_sev_ready = True
                any_loaded = True
            except Exception as e:
                print(f"  [ML Service] Error loading Severity: {e}")

        # Load metrics & status JSON
        metrics_file = self.artifacts_dir / "metrics.json"
        if metrics_file.exists():
            try:
                with open(metrics_file, "r", encoding="utf-8") as f:
                    self.metrics = json.load(f)
            except Exception:
                pass

        status_file = self.artifacts_dir / "status.json"
        if status_file.exists():
            try:
                with open(status_file, "r", encoding="utf-8") as f:
                    self.status = json.load(f)
            except Exception:
                pass

        return any_loaded

    @property
    def is_trained(self) -> bool:
        return self.m1_ready or self.m3_ready

    def get_status(self) -> Dict[str, Any]:
        """Provides status for tests, backend API, and React frontend."""
        is_tr = self.is_trained
        status_str = "TRAINED" if is_tr else "NOT_TRAINED"

        return {
            "status": status_str,
            "is_trained": is_tr,
            "overall_status": status_str,
            "dataset_mode": self.metrics.get("dataset_mode", "REAL_DATA" if is_tr else "UNKNOWN"),
            "trained_at": self.status.get("trained_at"),
            "training_duration_seconds": self.status.get("training_duration_seconds"),
            "total_training_rows": self.status.get("total_rows_used"),
            "models": {
                "m1_priority": {
                    "ready": self.m1_ready,
                    "status": "TRAINED" if self.m1_ready else "NOT_TRAINED",
                    "description": "Maintenance Priority (XGBoost Regressor -> risk_score)",
                },
                "m2_duration": {
                    "ready": False,
                    "status": "DATA_NOT_AVAILABLE",
                    "description": "Maintenance Duration Target Column Not Present in Datasets",
                    "note": "Architecture ready. Supply work-order duration data to enable.",
                },
                "m3_failure": {
                    "ready": self.m3_ready,
                    "status": "TRAINED" if self.m3_ready else "NOT_TRAINED",
                    "description": "Failure Risk (XGBoost Classifier -> maintenance_required)",
                },
                "m3_severity": {
                    "ready": self.m3_sev_ready,
                    "status": "TRAINED" if self.m3_sev_ready else "NOT_TRAINED",
                    "description": "Failure Severity (XGBoost Multiclass -> Low/Medium/High/Critical)",
                },
            },
            "explainability": "SHAP TreeExplainer" if is_tr else "DISABLED",
        }

    def _prepare_input(
        self,
        row: Dict[str, Any],
        feature_list: List[str],
        cat_cats: Dict[str, List[str]],
    ) -> pd.DataFrame:
        df = pd.DataFrame([row])

        # Fill missing features
        for col in feature_list:
            if col not in df.columns:
                if col in cat_cats:
                    df[col] = cat_cats[col][0] if cat_cats[col] else "Unknown"
                else:
                    df[col] = 0.0

        df = df[feature_list]

        # Categorical formatting
        for col, cats in cat_cats.items():
            if col in df.columns:
                val = str(df[col].iloc[0])
                if val not in cats:
                    val = cats[0] if cats else "Unknown"
                df[col] = pd.Categorical([val], categories=cats)

        # Numeric coercion
        for col in feature_list:
            if col not in cat_cats and col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        return df

    def predict(self, input_row: Dict[str, Any]) -> Dict[str, Any]:
        """Runs live inference across all trained models."""
        if not self.is_trained:
            return {
                "status": "MODEL_NOT_TRAINED",
                "message": "No trained models found. Run: python train_all_models.py",
            }

        res: Dict[str, Any] = {"status": "SUCCESS"}

        # 1. M1 Priority
        if self.m1_ready:
            X1 = self._prepare_input(input_row, self.m1_features, self.m1_cat_cats)
            raw = float(self.m1.predict(X1)[0])
            score = float(np.clip(raw, 0.0, 100.0))
            level = "CRITICAL" if score >= 75 else "HIGH" if score >= 55 else "MEDIUM" if score >= 35 else "LOW"
            res["priority"] = {
                "score": round(score, 2),
                "level": level,
                "lower_bound": round(float(np.clip(score - 1.645 * self.m1_std_err, 0, 100)), 2),
                "upper_bound": round(float(np.clip(score + 1.645 * self.m1_std_err, 0, 100)), 2),
            }
        else:
            res["priority"] = {"status": "MODEL_NOT_TRAINED"}

        # 2. M2 Duration (not available in datasets)
        res["duration"] = {
            "status": "DATA_NOT_AVAILABLE",
            "message": "Maintenance duration target column not in dataset",
            "predicted_minutes": int(input_row.get("nominal_duration_minutes", 90)),
        }

        # 3. M3 Failure Risk
        if self.m3_ready:
            X3 = self._prepare_input(input_row, self.m3_features, self.m3_cat_cats)
            probs = self.m3.predict_proba(X3)[0]
            fail_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
            risk_level = "CRITICAL" if fail_prob >= 0.70 else "HIGH" if fail_prob >= 0.45 else "MEDIUM" if fail_prob >= 0.20 else "LOW"
            res["failure_risk"] = {
                "probability": round(fail_prob, 4),
                "level": risk_level,
            }
        else:
            res["failure_risk"] = {"status": "MODEL_NOT_TRAINED"}

        # 4. M3 Failure Severity
        if self.m3_sev_ready:
            try:
                X_sev = self._prepare_input(input_row, self.m3_sev_features, self.m3_sev_cat_cats)
                sev_idx = int(self.m3_sev.predict(X_sev)[0])
                sev_label = (
                    self.m3_sev_label_classes[sev_idx]
                    if sev_idx < len(self.m3_sev_label_classes)
                    else "Unknown"
                )
                res["failure_severity"] = {"predicted_severity": sev_label}
            except Exception as e:
                res["failure_severity"] = {"status": "PREDICTION_ERROR", "detail": str(e)}
        else:
            res["failure_severity"] = {"status": "MODEL_NOT_TRAINED"}

        # 5. SHAP Explanation
        res["shap_explanation"] = self._explain(input_row)
        res["shap_reasons"] = res["shap_explanation"].get("top_reasons", ["Operational telemetry evaluation"])

        return res

    def predict_task(self, task: Dict[str, Any], section_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Translates maintenance task attributes and section telemetry into model inputs,
        then produces real ML predictions for the AI Decision Engine and OR-Tools scheduler.
        """
        sec = section_info or {}
        input_row = {
            "train_age_years": float(task.get("train_age_years", 14)),
            "average_speed_kmph": float(task.get("average_speed_kmph", 80)),
            "distance_travelled_km": float(task.get("distance_travelled_km", 95000)),
            "wheel_wear_percent": float(task.get("wheel_wear_percent", task.get("wear_percentage", 45))),
            "track_vibration_level": float(task.get("track_vibration_level", sec.get("avg_vibration", 3.2))),
            "rail_wear_mm": float(task.get("rail_wear_mm", sec.get("rail_wear_mm", 8.5))),
            "bearing_temperature_c": float(task.get("bearing_temperature_c", 78)),
            "axle_temperature_c": float(task.get("axle_temperature_c", 65)),
            "brake_pad_wear_percent": float(task.get("brake_pad_wear_percent", 48)),
            "brake_pressure_psi": float(task.get("brake_pressure_psi", 92)),
            "battery_voltage": float(task.get("battery_voltage", 24.1)),
            "last_maintenance_days": float(task.get("last_maintenance_days", 140)),
            "sensor_health_index": float(task.get("sensor_health_index", 70)),
            "inspection_score": float(task.get("inspection_score", 72)),
            "delay_minutes": float(task.get("delay_minutes", 10)),
            "ambient_temperature_c": float(sec.get("temperature_c", 30)),
            "humidity_percent": float(sec.get("humidity_percent", 60)),
            "rainfall_mm": float(sec.get("rainfall_mm", 5)),
            "region": task.get("region", sec.get("region", "Northern Railway")),
            "season": "Summer",
            "train_type": task.get("train_type", "Passenger"),
            "nominal_duration_minutes": task.get("nominal_duration_minutes", 90),
        }
        return self.predict(input_row)

    def _explain(self, input_row: Dict[str, Any]) -> Dict[str, Any]:
        """Generates SHAP attribution for M1 using native XGBoost TreeSHAP."""
        if not self.m1_ready or not self.m1_features:
            return {"top_factors": [], "top_reasons": []}
        try:
            import xgboost as xgb
            X1 = self._prepare_input(input_row, self.m1_features, self.m1_cat_cats)
            X_enc = X1.copy()
            for col in X_enc.select_dtypes(include="category").columns:
                X_enc[col] = X_enc[col].cat.codes

            booster = self.m1.get_booster() if hasattr(self.m1, "get_booster") else self.m1
            dmat = xgb.DMatrix(X_enc)
            contribs = booster.predict(dmat, pred_contribs=True)

            if contribs.ndim == 2:
                sv_arr = contribs[0, :-1]
            else:
                sv_arr = contribs[:-1]

            feats = self.m1_features
            pairs = sorted(zip(feats, sv_arr), key=lambda x: abs(x[1]), reverse=True)
            top = pairs[:6]
            factors = [
                {
                    "feature": f,
                    "shap_value": round(float(v), 4),
                    "direction": "increases_priority" if v > 0 else "decreases_priority",
                }
                for f, v in top
            ]
            reasons = [
                f"{f.replace('_', ' ').title()} ({'elevated' if v > 0 else 'nominal'})"
                for f, v in top[:3]
            ]
            return {"top_factors": factors, "top_reasons": reasons}
        except Exception as e:
            return {"error": str(e), "top_factors": [], "top_reasons": []}
