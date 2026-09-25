import os
from pathlib import Path
from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np

class RailwayPreprocessor:
    """
    Data normalization and railway network reconstruction engine.
    Derives section connectivity, train movements, and headway analytics.
    """

    def __init__(self, processed_dir: str = "data/processed"):
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def clean_stations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans station data, formats coordinates, and removes invalid entries."""
        clean = df.copy()
        clean = clean.dropna(subset=["code"])
        clean["code"] = clean["code"].astype(str).str.strip().str.upper()
        clean["name"] = clean["name"].fillna("Unknown Station").str.strip()
        clean["state"] = clean["state"].fillna("Unknown")
        clean["zone"] = clean["zone"].fillna("Unknown")
        clean["latitude"] = pd.to_numeric(clean["latitude"], errors="coerce")
        clean["longitude"] = pd.to_numeric(clean["longitude"], errors="coerce")
        clean = clean.drop_duplicates(subset=["code"])
        return clean

    def clean_trains(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans train records, standardizes IDs and types."""
        clean = df.copy()
        clean["train_number"] = clean["train_number"].astype(str).str.strip().str.zfill(5)
        clean["name"] = clean["name"].fillna("Express Special").str.strip()
        clean["type"] = clean["type"].fillna("EXPRESS").str.strip().str.upper()
        clean["from_station_code"] = clean["from_station_code"].astype(str).str.strip().str.upper()
        clean["to_station_code"] = clean["to_station_code"].astype(str).str.strip().str.upper()
        clean["distance"] = pd.to_numeric(clean.get("distance", 0), errors="coerce").fillna(0)
        clean = clean.drop_duplicates(subset=["train_number"])
        return clean

    def clean_schedules(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardizes timetable schedule stops."""
        clean = df.copy()
        clean["train_number"] = clean["train_number"].astype(str).str.strip().str.zfill(5)
        clean["station_code"] = clean["station_code"].astype(str).str.strip().str.upper()
        clean["station_name"] = clean.get("station_name", "").fillna("").astype(str).str.strip()
        
        # Replace 'None' string with NaN
        clean["arrival"] = clean["arrival"].replace("None", np.nan)
        clean["departure"] = clean["departure"].replace("None", np.nan)
        
        # Ensure day sequence is integer
        clean["day"] = pd.to_numeric(clean.get("day", 1), errors="coerce").fillna(1).astype(int)
        
        # Sort by train and timetable sequence id
        if "id" in clean.columns:
            clean = clean.sort_values(by=["train_number", "id"])
        
        return clean

    def build_railway_sections(self, schedules_df: pd.DataFrame, trains_df: pd.DataFrame) -> pd.DataFrame:
        """
        Reconstructs consecutive station-to-station sections (e.g. MAS-AJJ, AJJ-KPD)
        and computes operational traffic density, headway, and peak periods.
        """
        print("Reconstructing railway sections from schedule stops...")
        sorted_sch = schedules_df.copy()
        if "id" in sorted_sch.columns:
            sorted_sch = sorted_sch.sort_values(by=["train_number", "id"])
        
        # Group by train to identify consecutive station transitions
        sorted_sch["next_station"] = sorted_sch.groupby("train_number")["station_code"].shift(-1)
        sorted_sch["next_arrival"] = sorted_sch.groupby("train_number")["arrival"].shift(-1)
        
        # Filter transitions where next_station exists
        transitions = sorted_sch.dropna(subset=["next_station"]).copy()
        transitions = transitions[transitions["station_code"] != transitions["next_station"]]
        
        # Canonical section key: vectorized min/max
        st_a = transitions["station_code"].astype(str)
        st_b = transitions["next_station"].astype(str)
        cond = st_a <= st_b
        transitions["station_from"] = np.where(cond, st_a, st_b)
        transitions["station_to"] = np.where(cond, st_b, st_a)
        transitions["section_id"] = transitions["station_from"] + "-" + transitions["station_to"]
        
        # Merge train type information
        train_types = trains_df.set_index("train_number")["type"].to_dict()
        transitions["train_type"] = transitions["train_number"].map(train_types).fillna("EXPRESS")

        # Aggregate section statistics
        def parse_dep_hour(t):
            if pd.isna(t) or str(t) == "None" or not str(t):
                return 12
            try:
                return int(str(t).split(":")[0])
            except Exception:
                return 12

        transitions["dep_hour"] = transitions["departure"].apply(parse_dep_hour)

        sections_list = []
        for (sec_id, st1, st2), grp in transitions.groupby(["section_id", "station_from", "station_to"]):
            total_trains = len(grp["train_number"].unique())
            if total_trains < 1:
                continue

            # Trains per day estimation
            trains_per_day = max(total_trains, 1)
            
            # Category breakdown
            types = grp["train_type"].value_counts()
            passenger_count = int(types.get("PASSENGER", 0) + types.get("MEMU", 0) + types.get("DEMU", 0))
            express_count = int(total_trains - passenger_count)
            goods_frequency = max(1, int(total_trains * 0.15)) # 15% estimated freight traffic

            # Headway calculation (in minutes) across 24h operational day
            active_hours = 20 # active operating window
            avg_headway_min = round((active_hours * 60) / max(total_trains, 1), 1)
            min_headway_min = max(6, int(avg_headway_min * 0.35))

            # Peak hour determination
            hour_counts = grp["dep_hour"].value_counts()
            peak_hour = int(hour_counts.index[0]) if not hour_counts.empty else 8
            peak_window = f"{peak_hour:02d}:00-{(peak_hour+2)%24:02d}:00"
            off_peak_window = "01:00-04:30" if avg_headway_min < 30 else "11:00-14:00"

            # Traffic density tier
            if trains_per_day > 60:
                density = "VERY_HIGH"
                sensitivity = "CRITICAL"
            elif trains_per_day > 35:
                density = "HIGH"
                sensitivity = "HIGH"
            elif trains_per_day > 15:
                density = "MEDIUM"
                sensitivity = "MODERATE"
            else:
                density = "LOW"
                sensitivity = "STANDARD"

            sections_list.append({
                "section_id": sec_id,
                "station_from": st1,
                "station_to": st2,
                "trains_per_day": trains_per_day,
                "traffic_density": density,
                "avg_headway_minutes": avg_headway_min,
                "min_headway_minutes": min_headway_min,
                "passenger_frequency": passenger_count,
                "express_frequency": express_count,
                "goods_frequency": goods_frequency,
                "peak_window": peak_window,
                "off_peak_window": off_peak_window,
                "operational_sensitivity": sensitivity
            })

        sections_df = pd.DataFrame(sections_list)
        return sections_df

    def build_station_analytics(self, schedules_df: pd.DataFrame, stations_df: pd.DataFrame) -> pd.DataFrame:
        """Computes live station operational analytics: arrivals, departures, connected lines."""
        print("Computing station-level operational analytics...")
        sch = schedules_df.copy()
        
        arr_counts = sch[sch["arrival"].notna() & (sch["arrival"] != "None")].groupby("station_code")["train_number"].nunique()
        dep_counts = sch[sch["departure"].notna() & (sch["departure"] != "None")].groupby("station_code")["train_number"].nunique()
        total_counts = sch.groupby("station_code")["train_number"].nunique()

        stats = []
        for code, total in total_counts.items():
            arrivals = int(arr_counts.get(code, 0))
            departures = int(dep_counts.get(code, 0))
            
            # Headway
            avg_hw = round((18 * 60) / max(total, 1), 1)
            
            stats.append({
                "station_code": code,
                "total_trains": int(total),
                "arrivals": arrivals,
                "departures": departures,
                "avg_headway_minutes": avg_hw,
                "peak_period": "07:00-10:30, 17:30-20:30" if total > 30 else "08:00-11:00",
                "off_peak_period": "00:30-04:30" if total > 50 else "12:00-15:00"
            })

        stats_df = pd.DataFrame(stats)
        merged = stations_df.merge(stats_df, left_on="code", right_on="station_code", how="left")
        merged["total_trains"] = merged["total_trains"].fillna(0).astype(int)
        merged["arrivals"] = merged["arrivals"].fillna(0).astype(int)
        merged["departures"] = merged["departures"].fillna(0).astype(int)
        return merged

    def run_pipeline(self, stations_raw: pd.DataFrame, trains_raw: pd.DataFrame, schedules_raw: pd.DataFrame) -> Dict[str, Any]:
        """Runs complete normalization and saves parquet artifacts."""
        clean_st = self.clean_stations(stations_raw)
        clean_tr = self.clean_trains(trains_raw)
        clean_sc = self.clean_schedules(schedules_raw)

        sections_df = self.build_railway_sections(clean_sc, clean_tr)
        station_stats_df = self.build_station_analytics(clean_sc, clean_st)

        # Save to Parquet and JSON
        clean_st.to_parquet(self.processed_dir / "stations.parquet", index=False)
        clean_tr.to_parquet(self.processed_dir / "trains.parquet", index=False)
        clean_sc.to_parquet(self.processed_dir / "schedules.parquet", index=False)
        sections_df.to_parquet(self.processed_dir / "sections.parquet", index=False)
        station_stats_df.to_parquet(self.processed_dir / "station_analytics.parquet", index=False)

        # Also save sample JSON for fast frontend API access
        sections_df.head(500).to_json(self.processed_dir / "sections_sample.json", orient="records", indent=2)
        station_stats_df.head(500).to_json(self.processed_dir / "stations_sample.json", orient="records", indent=2)

        return {
            "stations_processed": len(clean_st),
            "trains_processed": len(clean_tr),
            "schedules_processed": len(clean_sc),
            "sections_reconstructed": len(sections_df),
            "station_analytics_count": len(station_stats_df)
        }
