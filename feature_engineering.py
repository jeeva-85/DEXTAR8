import os
import json
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

class FeatureEngineer:
    """
    Feature engineering and dataset cross-join engine.
    Combines maintenance tasks, physical asset characteristics, and real-time
    reconstructed railway section traffic density into ML feature sets.
    """

    CATEGORICAL_COLS = [
        "department", "asset_type", "maintenance_type", "machine_required",
        "traffic_density", "operational_sensitivity"
    ]

    NUMERIC_COLS = [
        "criticality", "defect_severity", "asset_age", "gmt", "days_overdue",
        "previous_deferrals", "condition_score", "failure_history", "crew_required",
        "trains_per_day", "avg_headway_minutes", "min_headway_minutes",
        "goods_frequency", "passenger_frequency", "express_frequency"
    ]

    def __init__(self, processed_dir: str = "data/processed", artifacts_dir: str = "ml/artifacts"):
        self.processed_dir = Path(processed_dir)
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def merge_maintenance_with_sections(self, maintenance_df: pd.DataFrame, sections_df: pd.DataFrame) -> pd.DataFrame:
        """Joins maintenance tasks with corresponding railway section traffic metrics."""
        sec_features = sections_df[[
            "section_id", "trains_per_day", "traffic_density", "avg_headway_minutes",
            "min_headway_minutes", "passenger_frequency", "express_frequency",
            "goods_frequency", "operational_sensitivity"
        ]].copy()

        # Left join on section_id
        merged = maintenance_df.merge(sec_features, left_on="section", right_on="section_id", how="left")
        
        # Fill sections without traffic stats with conservative mainline defaults
        merged["trains_per_day"] = merged["trains_per_day"].fillna(30.0)
        merged["traffic_density"] = merged["traffic_density"].fillna("MEDIUM")
        merged["avg_headway_minutes"] = merged["avg_headway_minutes"].fillna(35.0)
        merged["min_headway_minutes"] = merged["min_headway_minutes"].fillna(12.0)
        merged["passenger_frequency"] = merged["passenger_frequency"].fillna(15.0)
        merged["express_frequency"] = merged["express_frequency"].fillna(12.0)
        merged["goods_frequency"] = merged["goods_frequency"].fillna(3.0)
        merged["operational_sensitivity"] = merged["operational_sensitivity"].fillna("MODERATE")

        return merged

    def prepare_ml_datasets(self, merged_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Prepares training datasets for M1 (priority), M2 (duration), and M3 (failure risk)."""
        feature_cols = self.NUMERIC_COLS + self.CATEGORICAL_COLS

        # Encode categorical columns as category dtype for XGBoost native categorical support
        df_encoded = merged_df.copy()
        cat_categories = {}
        for col in self.CATEGORICAL_COLS:
            df_encoded[col] = df_encoded[col].astype(str)
            cats = sorted(df_encoded[col].unique().tolist())
            cat_categories[col] = cats
            df_encoded[col] = pd.Categorical(df_encoded[col], categories=cats)

        # M1 Dataset
        m1_df = df_encoded[feature_cols + ["priority_score", "task_id"]].copy()
        # M2 Dataset
        m2_df = df_encoded[feature_cols + ["actual_duration_minutes", "task_id"]].copy()
        # M3 Dataset
        m3_df = df_encoded[feature_cols + ["observed_failure", "task_id"]].copy()

        # Save Parquets
        m1_df.to_parquet(self.processed_dir / "ml_priority_dataset.parquet", index=False)
        m2_df.to_parquet(self.processed_dir / "ml_duration_dataset.parquet", index=False)
        m3_df.to_parquet(self.processed_dir / "ml_failure_dataset.parquet", index=False)

        # Save Feature Schema
        schema = {
            "feature_columns": feature_cols,
            "numeric_columns": self.NUMERIC_COLS,
            "categorical_columns": self.CATEGORICAL_COLS,
            "categories": cat_categories,
            "targets": {
                "m1": "priority_score",
                "m2": "actual_duration_minutes",
                "m3": "observed_failure"
            },
            "timestamp": pd.Timestamp.now().isoformat()
        }
        with open(self.artifacts_dir / "feature_schema.json", "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2)

        return {"m1": m1_df, "m2": m2_df, "m3": m3_df}

    def transform_single_task(self, task_dict: Dict[str, Any], section_traffic: Dict[str, Any]) -> pd.DataFrame:
        """Transforms a single maintenance task request into model input dataframe."""
        row = {**task_dict, **section_traffic}
        df = pd.DataFrame([row])

        # Load categories from schema if exists
        schema_path = self.artifacts_dir / "feature_schema.json"
        cat_map = {}
        if schema_path.exists():
            try:
                with open(schema_path, "r", encoding="utf-8") as f:
                    cat_map = json.load(f).get("categories", {})
            except Exception:
                pass

        for col in self.NUMERIC_COLS:
            if col not in df.columns:
                df[col] = 0.0
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        for col in self.CATEGORICAL_COLS:
            val = str(df[col].iloc[0]) if col in df.columns and pd.notna(df[col].iloc[0]) else None
            known = cat_map.get(col, [])
            if known:
                if val not in known:
                    val = known[0]
                df[col] = pd.Categorical([val], categories=known)
            else:
                df[col] = pd.Categorical([val])

        feature_cols = self.NUMERIC_COLS + self.CATEGORICAL_COLS
        return df[feature_cols]
