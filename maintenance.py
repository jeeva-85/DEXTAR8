from fastapi import APIRouter, Query
from typing import Optional
from backend.services.data_service import data_service

router = APIRouter(prefix="/api/maintenance", tags=["Maintenance"])

@router.get("")
def list_maintenance_tasks(
    department: Optional[str] = None,
    limit: int = Query(default=100, le=500)
):
    tasks = data_service.get_maintenance_tasks(department=department, limit=limit)
    return {
        "count": len(tasks),
        "department_filter": department or "ALL",
        "mode": "SIMULATION MODE" if tasks and tasks[0].get("is_simulated") else "OPERATIONAL",
        "tasks": tasks
    }
