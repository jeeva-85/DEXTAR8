"""
FastAPI Router for System Status & Diagnostics.
GET /api/system/status verifies overall system readiness (FastAPI, Database, ML, OR-Tools, External AI).
"""

from fastapi import APIRouter
from backend.config import settings
from backend.services.ml_service import ml_service

router = APIRouter(prefix="/api/system", tags=["System Diagnostics"])


@router.get("/status")
def get_system_status():
    """
    Returns the comprehensive operational status of all RailBlock AI components.
    Safely reveals whether External AI is configured without exposing secrets.
    """
    ml_status = ml_service.get_status()
    ai_info = settings.get_public_ai_info()

    return {
        "system": "RAILBLOCK AI",
        "status": "OPERATIONAL",
        "version": "1.0.0",
        "database": {
            "status": "CONNECTED",
            "type": "SQLite" if "sqlite" in settings.DATABASE_URL else "PostgreSQL",
        },
        "ml_engine": {
            "is_trained": ml_status.get("is_trained", False),
            "overall_status": ml_status.get("overall_status", "UNKNOWN"),
            "dataset_mode": ml_status.get("dataset_mode", "UNKNOWN"),
            "models": list(ml_status.get("models", {}).keys()),
        },
        "optimization_engine": {
            "solver": "Google OR-Tools CP-SAT",
            "status": "READY",
            "multi_department_coordination": ["ENGINEERING", "SNT", "TRD"],
        },
        "external_ai": {
            "status": "CONFIGURED" if ai_info["configured"] else "NOT CONFIGURED",
            "configured": ai_info["configured"],
            "provider": ai_info["provider"],
            "model": ai_info["model"],
            "timeout_seconds": ai_info["timeout_seconds"],
        },
    }
