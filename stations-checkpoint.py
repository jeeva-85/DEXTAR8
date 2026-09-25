from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from backend.services.data_service import data_service

router = APIRouter(prefix="/api/stations", tags=["Stations"])

@router.get("")
def list_stations(limit: int = Query(default=100, le=500), search: Optional[str] = None):
    stations = data_service.get_stations(limit=500)
    if search:
        s = search.upper()
        stations = [st for st in stations if s in str(st.get("code", "")) or s in str(st.get("name", "")).upper()]
    return stations[:limit]

@router.get("/{station_code}")
def get_station_details(station_code: str):
    st = data_service.get_station_by_code(station_code)
    if not st:
        raise HTTPException(status_code=404, detail=f"Station code '{station_code}' not found")
    return st

@router.get("/{station_code}/traffic")
def get_station_traffic(station_code: str):
    st = data_service.get_station_by_code(station_code)
    if not st:
        raise HTTPException(status_code=404, detail=f"Station code '{station_code}' not found")
    return {
        "station_code": st.get("code"),
        "name": st.get("name"),
        "total_trains": st.get("total_trains", 0),
        "arrivals": st.get("arrivals", 0),
        "departures": st.get("departures", 0),
        "avg_headway_minutes": st.get("avg_headway_minutes", 0),
        "peak_period": st.get("peak_period", "08:00-11:00"),
        "off_peak_period": st.get("off_peak_period", "12:00-15:00")
    }
