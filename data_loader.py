import os
import json
import zipfile
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import pandas as pd

class RailwayDataLoader:
    """
    Robust data loader capable of discovering, extracting, and loading
    raw railway datasets (JSON, CSV, Parquet, ZIP).
    """

    def __init__(self, raw_dir: str = "data/raw", extracted_dir: str = "data/extracted"):
        self.raw_dir = Path(raw_dir)
        self.extracted_dir = Path(extracted_dir)
        self.extracted_dir.mkdir(parents=True, exist_ok=True)

    def discover_files(self) -> Dict[str, list]:
        """Discovers all supported data archives and structured files."""
        inventory = {
            "zip": list(self.raw_dir.glob("*.zip")),
            "json": list(self.raw_dir.glob("*.json")),
            "csv": list(self.raw_dir.glob("*.csv")),
            "parquet": list(self.raw_dir.glob("*.parquet")),
        }
        return inventory

    def extract_zip_archives(self):
        """Extracts any ZIP files found in data/raw without modifying originals."""
        for zip_path in self.raw_dir.glob("*.zip"):
            extract_target = self.extracted_dir / zip_path.stem
            if not extract_target.exists():
                print(f"Extracting {zip_path.name} to {extract_target}...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_target)

    def load_stations_raw(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Loads GeoJSON stations.json and normalizes to tabular DataFrame."""
        stations_path = self.raw_dir / "stations.json"
        if not stations_path.exists():
            raise FileNotFoundError(f"Missing stations file: {stations_path}")

        with open(stations_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        records = []
        for feat in data.get("features", []):
            geom = feat.get("geometry") or {}
            coords = geom.get("coordinates") if geom else None
            lon, lat = (coords[0], coords[1]) if coords and len(coords) >= 2 else (None, None)
            props = feat.get("properties") or {}
            records.append({
                "code": props.get("code"),
                "name": props.get("name"),
                "state": props.get("state"),
                "zone": props.get("zone"),
                "address": props.get("address"),
                "longitude": lon,
                "latitude": lat
            })
        df = pd.DataFrame(records)
        meta = {
            "raw_count": len(df),
            "columns": list(df.columns),
            "file_size_bytes": os.path.getsize(stations_path)
        }
        return df, meta

    def load_trains_raw(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Loads GeoJSON trains.json and extracts train properties and geometry."""
        trains_path = self.raw_dir / "trains.json"
        if not trains_path.exists():
            raise FileNotFoundError(f"Missing trains file: {trains_path}")

        with open(trains_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        records = []
        for feat in data.get("features", []):
            props = feat.get("properties") or {}
            geom = feat.get("geometry") or {}
            coords = geom.get("coordinates") if geom else []
            props_copy = dict(props)
            props_copy["route_points_count"] = len(coords) if coords else 0
            # Normalize train identifier
            props_copy["train_number"] = str(props.get("number", "")).zfill(5)
            records.append(props_copy)

        df = pd.DataFrame(records)
        meta = {
            "raw_count": len(df),
            "columns": list(df.columns),
            "file_size_bytes": os.path.getsize(trains_path)
        }
        return df, meta

    def load_schedules_raw(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Loads schedules.json timetable records."""
        schedules_path = self.raw_dir / "schedules.json"
        if not schedules_path.exists():
            raise FileNotFoundError(f"Missing schedules file: {schedules_path}")

        with open(schedules_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        df = pd.DataFrame(data)
        if "train_number" in df.columns:
            df["train_number"] = df["train_number"].astype(str).str.zfill(5)

        meta = {
            "raw_count": len(df),
            "columns": list(df.columns),
            "file_size_bytes": os.path.getsize(schedules_path)
        }
        return df, meta

    def get_inventory_summary(self) -> Dict[str, Any]:
        """Produces a comprehensive metadata inventory of all detected datasets."""
        inventory = self.discover_files()
        summary = {"files": {}}
        for category, paths in inventory.items():
            for p in paths:
                summary["files"][p.name] = {
                    "format": p.suffix.lstrip("."),
                    "size_bytes": os.path.getsize(p),
                    "size_mb": round(os.path.getsize(p) / (1024 * 1024), 2),
                    "path": str(p)
                }
        return summary
