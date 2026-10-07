"""Query endpoint: semantic is real; other modes use the shared run_pattern placeholder."""

from fastapi import APIRouter, HTTPException

from building_with_rag.contracts import QueryRequest, QueryResult
from building_with_rag.registry import Pattern, run_pattern
from building_with_rag.retrieval.semantic import RetrievalError, semantic_retrieve

router = APIRouter()


@router.post("/v1/query")
def query(request: QueryRequest) -> QueryResult:
    if request.pattern is Pattern.SEMANTIC:
        try:
            return semantic_retrieve(request)
        except RetrievalError as exc:
            raise HTTPException(
                status_code=exc.status_code,
                detail={"error": {"message": exc.message, "code": exc.code}},
            ) from None
    payload = run_pattern(request.pattern, request.question, request.caller_id)
    return QueryResult(**payload)
