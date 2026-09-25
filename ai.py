"""
FastAPI Router for External AI Assistant & Explanation Layer.
Provides /api/ai/analyze endpoint to explain ML predictions, maintenance urgency,
schedules, and what-if simulations.
"""

import logging
from fastapi import APIRouter, HTTPException, status
from backend.schemas.ai import AIAnalysisRequest, AIAnalysisResponse
from backend.services.ai_context_builder import AIContextBuilder
from backend.services.external_ai_service import external_ai_service

logger = logging.getLogger("railblock.api.ai")

router = APIRouter(prefix="/api/ai", tags=["AI Assistant & Explanation"])


@router.post(
    "/analyze",
    response_model=AIAnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Explain RailBlock AI predictions, block schedules, or scenarios using External AI",
)
def analyze_with_ai(req: AIAnalysisRequest) -> AIAnalysisResponse:
    """
    1. Validates request
    2. Builds factual RailBlock AI context from live services
    3. Sends sanitized prompt to external AI service
    4. Validates and returns structured explanation to frontend
    """
    try:
        controlled_context, user_prompt, system_instruction = AIContextBuilder.build_context(
            action_type=req.action_type,
            question=req.question,
            user_context=req.context,
        )

        result = external_ai_service.generate_explanation(
            prompt=user_prompt,
            system_instruction=system_instruction,
            context_type=req.action_type,
        )

        return AIAnalysisResponse(
            success=result["success"],
            answer=result["answer"],
            source=result.get("source", "external_ai"),
            context_type=result.get("context_type", req.action_type),
            provider=result.get("provider"),
            model=result.get("model"),
            error_code=result.get("error_code"),
            controlled_context=controlled_context,
        )
    except Exception as e:
        logger.error("Unexpected error in /api/ai/analyze: %s", type(e).__name__)
        return AIAnalysisResponse(
            success=False,
            answer="Unable to process the AI request.",
            source="external_ai",
            context_type=req.action_type,
            error_code="INTERNAL_ERROR",
            controlled_context=None,
        )
