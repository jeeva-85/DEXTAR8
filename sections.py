from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from backend.services.data_service import data_service

router = APIRouter(prefix="/api/sections", tags=["Sections"])

@router.get("")
def list_sections(limit: int = Query(default=100, le=500)):
    return data_service.get_sections(limit=limit)

@router.get("/{section_id}")
def get_section_details(section_id: str):
    sec = data_service.get_section_by_id(section_id)
    if not sec:
        raise HTTPException(status_code=404, detail=f"Section '{section_id}' not found")
    return sec
