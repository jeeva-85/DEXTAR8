from fastapi import APIRouter

router = APIRouter(prefix="/api/conflicts", tags=["Conflicts"])

@router.get("")
def list_conflicts():
    return {
        "active_conflicts_count": 0,
        "resolved_conflicts_count": 8,
        "conflicts": [
            {
                "conflict_id": "CONF-RESOLVED-101",
                "section_id": "MAS-AJJ",
                "train_number": "12601",
                "severity": "HIGH",
                "description": "Potential clash between 120-min tamping block and Mangalore Mail departure. Resolved by shifting block to candidate window 01:00-04:30.",
                "status": "RESOLVED_BY_OR_TOOLS"
            }
        ]
    }
