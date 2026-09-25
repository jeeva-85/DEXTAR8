from fastapi import APIRouter
from backend.services.data_service import data_service

router = APIRouter(prefix="/api/analytics", tags=["Analytics"])

@router.get("/dashboard-kpis")
def get_dashboard_kpis():
    tasks = data_service.get_maintenance_tasks(limit=500)
    sections = data_service.get_sections(limit=100)
    
    crit_count = sum(1 for t in tasks if t.get("criticality", 0) >= 4)
    overdue_count = sum(1 for t in tasks if t.get("days_overdue", 0) > 0)
    avg_dur = round(sum(t.get("nominal_duration_minutes", 90) for t in tasks) / max(len(tasks), 1), 1)

    return {
        "critical_tasks": crit_count,
        "planned_blocks": 14,
        "coordinated_blocks": 9,
        "block_utilization_pct": 88.4,
        "asset_availability_pct": 96.2,
        "overdue_maintenance_count": overdue_count,
        "active_blocks_count": 1,
        "active_conflicts_count": 0,
        "average_block_duration_minutes": avg_dur,
        "optimization_score": 94.6,
        "total_sections_monitored": len(sections),
        "total_tasks_tracked": len(tasks)
    }
