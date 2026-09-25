import os
import json
from pathlib import Path
from typing import Dict, Any
import pandas as pd

class RailwayDataValidator:
    """
    Automated data quality validation engine.
    Checks integrity across stations, trains, and timetable schedules.
    """

    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def validate(self, stations_df: pd.DataFrame, trains_df: pd.DataFrame, schedules_df: pd.DataFrame) -> Dict[str, Any]:
        report = {
            "summary": {
                "stations_count": len(stations_df),
                "trains_count": len(trains_df),
                "schedules_count": len(schedules_df),
                "timestamp": pd.Timestamp.now().isoformat()
            },
            "stations_issues": self._validate_stations(stations_df),
            "trains_issues": self._validate_trains(trains_df),
            "schedules_issues": self._validate_schedules(schedules_df, stations_df, trains_df),
        }

        # Save JSON report
        json_path = self.reports_dir / "data_quality_report.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        # Save HTML report
        self._generate_html_report(report)

        return report

    def _validate_stations(self, df: pd.DataFrame) -> Dict[str, Any]:
        issues = {}
        issues["missing_code"] = int(df["code"].isna().sum()) if "code" in df.columns else 0
        issues["missing_coords"] = int((df["latitude"].isna() | df["longitude"].isna()).sum())
        issues["duplicate_codes"] = int(df["code"].duplicated().sum()) if "code" in df.columns else 0
        issues["invalid_latitudes"] = int(((df["latitude"] < -90) | (df["latitude"] > 90)).sum())
        issues["invalid_longitudes"] = int(((df["longitude"] < -180) | (df["longitude"] > 180)).sum())
        return issues

    def _validate_trains(self, df: pd.DataFrame) -> Dict[str, Any]:
        issues = {}
        issues["missing_number"] = int(df["train_number"].isna().sum()) if "train_number" in df.columns else 0
        issues["duplicate_train_numbers"] = int(df["train_number"].duplicated().sum()) if "train_number" in df.columns else 0
        if "distance" in df.columns:
            numeric_dist = pd.to_numeric(df["distance"], errors="coerce")
            issues["negative_distance"] = int((numeric_dist < 0).sum())
            issues["missing_distance"] = int(numeric_dist.isna().sum())
        return issues

    def _validate_schedules(self, df: pd.DataFrame, stations_df: pd.DataFrame, trains_df: pd.DataFrame) -> Dict[str, Any]:
        issues = {}
        issues["missing_arrival"] = int((df["arrival"].isna() | (df["arrival"] == "None")).sum())
        issues["missing_departure"] = int((df["departure"].isna() | (df["departure"] == "None")).sum())
        
        station_codes = set(stations_df["code"].dropna().unique())
        orphan_stations = (~df["station_code"].isin(station_codes)).sum()
        issues["orphan_station_references"] = int(orphan_stations)

        train_numbers = set(trains_df["train_number"].dropna().unique())
        orphan_trains = (~df["train_number"].isin(train_numbers)).sum()
        issues["orphan_train_references"] = int(orphan_trains)

        issues["duplicate_train_station_stop"] = int(df.duplicated(subset=["train_number", "station_code"]).sum())
        return issues

    def _generate_html_report(self, report: Dict[str, Any]):
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>RailBlock AI - Data Quality Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0b0f19; color: #e2e8f0; padding: 24px; }}
        h1 {{ color: #38bdf8; font-size: 24px; }}
        h2 {{ color: #94a3b8; font-size: 18px; margin-top: 20px; }}
        .card {{ background: #1e293b; border-radius: 8px; padding: 16px; margin-bottom: 16px; border: 1px solid #334155; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
        th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #334155; font-size: 14px; }}
        th {{ color: #38bdf8; font-weight: 600; }}
        .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-weight: bold; }}
        .badge-ok {{ background: #065f46; color: #34d399; }}
        .badge-warn {{ background: #854d0e; color: #fde047; }}
    </style>
</head>
<body>
    <h1>RailBlock AI — Data Quality Engine Report</h1>
    <div class="card">
        <h2>Ingestion Summary</h2>
        <p><strong>Timestamp:</strong> {report['summary']['timestamp']}</p>
        <p><strong>Total Stations:</strong> {report['summary']['stations_count']:,}</p>
        <p><strong>Total Train Services:</strong> {report['summary']['trains_count']:,}</p>
        <p><strong>Total Timetable Stops:</strong> {report['summary']['schedules_count']:,}</p>
    </div>

    <div class="card">
        <h2>Station Validation</h2>
        <table>
            <tr><th>Metric</th><th>Anomaly Count</th><th>Status</th></tr>
            {''.join(f"<tr><td>{k}</td><td>{v}</td><td><span class='badge " + ("badge-ok'>CLEAN" if v == 0 else "badge-warn'>DETECTED") + "</span></td></tr>" for k, v in report['stations_issues'].items())}
        </table>
    </div>

    <div class="card">
        <h2>Train Validation</h2>
        <table>
            <tr><th>Metric</th><th>Anomaly Count</th><th>Status</th></tr>
            {''.join(f"<tr><td>{k}</td><td>{v}</td><td><span class='badge " + ("badge-ok'>CLEAN" if v == 0 else "badge-warn'>DETECTED") + "</span></td></tr>" for k, v in report['trains_issues'].items())}
        </table>
    </div>

    <div class="card">
        <h2>Timetable Schedules Validation</h2>
        <table>
            <tr><th>Metric</th><th>Anomaly Count</th><th>Status</th></tr>
            {''.join(f"<tr><td>{k}</td><td>{v}</td><td><span class='badge " + ("badge-ok'>CLEAN" if v == 0 else "badge-warn'>DETECTED") + "</span></td></tr>" for k, v in report['schedules_issues'].items())}
        </table>
    </div>
</body>
</html>"""
        html_path = self.reports_dir / "data_quality_report.html"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
