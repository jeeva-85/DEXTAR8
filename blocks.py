from fastapi import APIRouter
from typing import List, Dict, Any

router = APIRouter(prefix="/api/blocks", tags=["Maintenance Blocks"])

# Dynamic in-memory state tracking for active block demo
ACTIVE_BLOCK_STATE = {
    "block_id": "BLK-MAS-AJJ-1000",
    "section_id": "MAS-AJJ",
    "status": "ACTIVE",
    "start_time": "10:00",
    "scheduled_end_time": "12:00",
    "elapsed_minutes": 75,
    "remaining_minutes": 45,
    "completion_percentage": 62.5,
    "is_coordinated": True,
    "coordination_type": "COORDINATED (ENG+S&T+TRD)",
    "departments": ["ENGINEERING", "SNT", "TRD"],
    "tasks": ["MT-1042", "MT-1088", "MT-1115"],
    "safety_status": "CORRIDOR_ISOLATED_SECURE"
}

@router.get("")
def list_blocks():
    return [
        ACTIVE_BLOCK_STATE,
        {
            "block_id": "BLK-AJJ-KPD-1115",
            "section_id": "AJJ-KPD",
            "status": "PLANNED",
            "start_time": "11:15",
            "scheduled_end_time": "13:30",
            "elapsed_minutes": 0,
            "remaining_minutes": 135,
            "completion_percentage": 0.0,
            "is_coordinated": True,
            "coordination_type": "COORDINATED (ENG+S&T)",
            "departments": ["ENGINEERING", "SNT"],
            "tasks": ["MT-1140", "MT-1152"],
            "safety_status": "AWAITING_FLAG_CLEARANCE"
        }
    ]

@router.get("/{block_id}")
def get_block(block_id: str):
    if block_id == ACTIVE_BLOCK_STATE["block_id"]:
        return ACTIVE_BLOCK_STATE
    return {
        "block_id": block_id,
        "status": "PLANNED",
        "section_id": "MAS-AJJ",
        "scheduled_end_time": "12:00",
        "elapsed_minutes": 0,
        "remaining_minutes": 120,
        "completion_percentage": 0.0
    }

@router.post("/{block_id}/progress")
def update_block_progress(block_id: str, progress_pct: float, elapsed_min: int):
    if block_id == ACTIVE_BLOCK_STATE["block_id"]:
        ACTIVE_BLOCK_STATE["completion_percentage"] = progress_pct
        ACTIVE_BLOCK_STATE["elapsed_minutes"] = elapsed_min
        ACTIVE_BLOCK_STATE["remaining_minutes"] = max(0, 120 - elapsed_min)
        if progress_pct >= 100.0:
            ACTIVE_BLOCK_STATE["status"] = "COMPLETED"
    return ACTIVE_BLOCK_STATE
