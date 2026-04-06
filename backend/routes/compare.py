"""
Standalone compare endpoint.

This route is intentionally independent from chat endpoints/state.
"""

from fastapi import APIRouter, HTTPException
from schemas.compare import CompareRequest, CompareResponse
from rag.compare_pipeline import run_compare_pipeline
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/compare", tags=["compare"])


@router.post("", response_model=CompareResponse)
async def compare_documents(request: CompareRequest):
    """
    Compare two selected documents for a focused query.
    """
    if len(request.doc_ids) < 2:
        raise HTTPException(status_code=400, detail="Please select at least two documents")

    try:
        result = await run_compare_pipeline(request.doc_ids, request.query, top_k=8)

        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error", "Compare failed"))

        return CompareResponse(
            summary=result.get("summary", {}),
            intent=result.get("intent", ""),
            docA=result.get("docA", {}),
            docB=result.get("docB", {}),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Compare endpoint failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to compare documents: {str(e)}")