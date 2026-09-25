from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from backend.services.data_service import data_service

router = APIRouter(prefix="/api/trains", tags=["Trains"])

@router.get("")
def list_trains(limit: int = Query(default=50, le=200)):
    return data_service.get_trains(limit=limit)
