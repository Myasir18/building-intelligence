"""Semantic retrieval: embed the question, run MongoDB $vectorSearch, resolve passages."""

import threading

from pymongo import MongoClient
from pymongo.errors import PyMongoError
from voyageai import Client as VoyageClient
from voyageai.error import VoyageError

from building_with_rag.contracts import QueryRequest, QueryResult, RetrievedChunk
from building_with_rag.ingestion import mongodb_schema as schema
from building_with_rag.settings import get_settings

QUERY_INPUT_TYPE = "query"
MIN_CANDIDATES = 50
MAX_CANDIDATES = 200
MONGO_TIMEOUT_MS = 5000
VOYAGE_TIMEOUT_S = 15

_lock = threading.Lock()
_mongo: MongoClient | None = None
_voyage: VoyageClient | None = None
_ready = False


class RetrievalError(Exception):
    def __init__(self, status_code: int, code: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def num_candidates(limit: int) -> int:
    """max(50, 10 * limit), capped at 200, never below limit."""
    return max(limit, min(MAX_CANDIDATES, max(MIN_CANDIDATES, limit * 10)))


def _not_ready(message: str) -> RetrievalError:
    return RetrievalError(503, "retrieval_not_ready", message)


def _upstream(message: str) -> RetrievalError:
    return RetrievalError(502, "retrieval_upstream_error", message)


def _bad_request(message: str) -> RetrievalError:
    return RetrievalError(422, "invalid_request", message)


def _clients():
    global _mongo, _voyage
    settings = get_settings()
    if not settings.mongodb_uri:
        raise _not_ready("MONGODB_URI is not set.")
    if not settings.voyage_api_key:
        raise _not_ready("VOYAGE_API_KEY is not set.")
    with _lock:
        if _mongo is None:
            _mongo = MongoClient(
                settings.mongodb_uri,
                serverSelectionTimeoutMS=MONGO_TIMEOUT_MS,
                connectTimeoutMS=MONGO_TIMEOUT_MS,
                socketTimeoutMS=VOYAGE_TIMEOUT_S * 1000,
            )
        if _voyage is None:
            _voyage = VoyageClient(
                api_key=settings.voyage_api_key, max_retries=0, timeout=VOYAGE_TIMEOUT_S
            )
    return _mongo[settings.mongodb_db_name], _voyage


def _ensure_ready(db) -> None:
    global _ready
    if _ready:
        return
    embeddings = db[schema.EMBEDDINGS_COLLECTION]
    try:
        indexes = list(embeddings.list_search_indexes(schema.VECTOR_INDEX_NAME))
        sample = embeddings.find_one({}, {"model": 1, "dimensions": 1})
    except PyMongoError:
        raise _upstream("MongoDB error while checking retrieval readiness.") from None
    if not indexes:
        raise _not_ready(
            f"Vector index '{schema.VECTOR_INDEX_NAME}' not found; run Story 2.2 ingestion."
        )
    if not indexes[0].get("queryable"):
        raise _not_ready(f"Vector index '{schema.VECTOR_INDEX_NAME}' is not queryable yet.")
    if sample is None:
        raise _not_ready("embeddings collection is empty; run Story 2.2 ingestion.")
    if (sample.get("model"), sample.get("dimensions")) != (
        schema.EMBEDDING_MODEL,
        schema.EMBEDDING_DIMENSIONS,
    ):
        raise _not_ready("Stored embeddings do not match the expected model/dimensions.")
    _ready = True


def _validate_scope(request: QueryRequest) -> dict[str, list[str]]:
    if request.caller_id is not None and request.caller_id != get_settings().webui_demo_caller_id:
        raise _bad_request("caller_id does not match the configured demo caller.")
    if request.required_acts is not None:
        raise _bad_request("required_acts is not supported by semantic mode.")
    if request.chapter is not None:
        raise _bad_request("chapter is not supported by semantic mode.")
    filters = request.filters
    return {
        "act": list(filters.act) if filters else [],
        "status": list(filters.status) if filters else [],
        # Server fixes access_level; caller values are validated to "public" and only narrow.
        "access_level": [schema.DEFAULT_ACCESS_LEVEL],
    }


def _filter_clause(scope: dict[str, list[str]]) -> dict:
    clauses = [{field: {"$in": values}} for field, values in scope.items() if values]
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def _search(request: QueryRequest, scope: dict[str, list[str]], candidates: int):
    global _ready
    try:
        db, voyage = _clients()
        _ensure_ready(db)
        try:
            vector = voyage.embed(
                texts=[request.question], model=schema.EMBEDDING_MODEL, input_type=QUERY_INPUT_TYPE
            ).embeddings[0]
        except VoyageError:
            raise _upstream("Voyage error while embedding the question.") from None
        if len(vector) != schema.EMBEDDING_DIMENSIONS:
            raise _not_ready(
                f"Voyage returned {len(vector)} dimensions, expected {schema.EMBEDDING_DIMENSIONS}."
            )
        try:
            hits = list(
                db[schema.EMBEDDINGS_COLLECTION].aggregate(
                    [
                        {
                            "$vectorSearch": {
                                "index": schema.VECTOR_INDEX_NAME,
                                "path": "vector",
                                "queryVector": vector,
                                "numCandidates": candidates,
                                "limit": request.limit,
                                "filter": _filter_clause(scope),
                            }
                        },
                        {
                            "$project": {
                                "_id": 0,
                                "chunk_id": 1,
                                "score": {"$meta": "vectorSearchScore"},
                            }
                        },
                    ]
                )
            )
            chunks = {
                c["_id"]: c
                for c in db[schema.CHUNKS_COLLECTION].find(
                    {"_id": {"$in": [h["chunk_id"] for h in hits]}}
                )
            }
            section_ids = list({c["section_id"] for c in chunks.values()})
            sections = {
                s["_id"]: s
                for s in db[schema.SECTIONS_COLLECTION].find({"_id": {"$in": section_ids}})
            }
        except PyMongoError:
            raise _upstream("MongoDB error during vector search.") from None
    except RetrievalError:
        _ready = False
        raise
    return hits, chunks, sections


def semantic_retrieve(request: QueryRequest) -> QueryResult:
    scope = _validate_scope(request)
    candidates = num_candidates(request.limit)
    hits, chunks, sections = _search(request, scope, candidates)

    results: list[RetrievedChunk] = []
    unresolved = 0
    for hit in hits:
        chunk = chunks.get(hit["chunk_id"])
        section = sections.get(chunk["section_id"]) if chunk else None
        if chunk is None or section is None:
            unresolved += 1
            continue
        results.append(
            RetrievedChunk(
                chunk_id=chunk["chunk_id"],
                section_id=section["section_id"],
                act=chunk["act"],
                text=chunk["text"],
                heading=section.get("heading") or "",
                score=float(hit["score"]),
                chunk_index=chunk.get("chunk_index"),
                act_label=section.get("act_label"),
                status=section.get("status"),
                chapter=section.get("chapter"),
                chapter_title=section.get("chapter_title"),
                section_number=section.get("section_number"),
                source_pdf=section.get("source_pdf"),
                source_sha256=section.get("source_sha256"),
                needs_review=section.get("needs_review"),
            )
        )

    ignored = (
        ["generate_answer: accepted, no answer is generated in semantic mode"]
        if request.generate_answer
        else []
    )
    if results:
        status = "ok"
        message = (
            f"{len(results)} passage(s) returned. Scores rank similarity only; "
            "they do not prove a passage is correct or answers the question."
        )
    else:
        status = "no_results"
        message = (
            "No passages match the given filters. Scores rank similarity only; "
            "they do not prove correctness."
        )
    return QueryResult(
        pattern="semantic",
        status=status,
        message=message,
        trace={
            "mode": "semantic",
            "query": request.question,
            "embedding": {
                "model": schema.EMBEDDING_MODEL,
                "input_type": QUERY_INPUT_TYPE,
                "dimensions": schema.EMBEDDING_DIMENSIONS,
            },
            "index": schema.VECTOR_INDEX_NAME,
            "limit": request.limit,
            "num_candidates": candidates,
            "filters": scope,
            "caller_id": get_settings().webui_demo_caller_id,
            "result_count": len(results),
            "ignored": ignored,
            "unresolved_hits": unresolved,
        },
        results=results,
    )
