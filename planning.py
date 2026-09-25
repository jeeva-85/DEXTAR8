from fastapi import APIRouter
from backend.schemas.planning import GeneratePlanRequest, OverrunEvaluationRequest
from backend.services.planning_service import planning_service
from optimization.reoptimization import DynamicBlockOverrunEngine

router = APIRouter(prefix="/api/planning", tags=["Planning & Optimization"])
overrun_engine = DynamicBlockOverrunEngine()

@router.post("/generate")
def generate_block_plan(req: GeneratePlanRequest):
    return planning_service.generate_plan(
        sections=req.sections,
        departments=req.departments,
        start_date=req.start_date,
        end_date=req.end_date
    )

@router.get("/weekly")
def get_weekly_plan():
    # Helper endpoint returning pre-calculated weekly schedule
    return planning_service.generate_plan(
        sections=["MAS-AJJ", "AJJ-KPD", "NDLS-CNB"],
        departments=["ENGINEERING", "SNT", "TRD"],
        start_date="2026-09-07",
        end_date="2026-09-13"
    )

@router.get("/monthly")
def get_monthly_plan():
    res = planning_service.generate_plan(
        sections=["MAS-AJJ", "AJJ-KPD", "KPD-SA", "NDLS-CNB", "CNB-PRYJ"],
        departments=["ENGINEERING", "SNT", "TRD"],
        start_date="2026-09-01",
        end_date="2026-09-30"
    )
    return {
        **res,
        "month": "September 2026",
        "total_monthly_backlog": 142,
        "critical_backlog": 18
    }

@router.post("/overrun-evaluation")
def evaluate_overrun(req: OverrunEvaluationRequest):
    return overrun_engine.evaluate_overrun(
        block_id=req.block_id,
        section_id=req.section_id,
        scheduled_end_time=req.scheduled_end_time,
        overrun_minutes=req.overrun_minutes
    )
