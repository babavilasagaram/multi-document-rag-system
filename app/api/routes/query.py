import re
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.retrieval.retriever import retrieve_documents
from app.generation.generator import generate_answer

from app.api.schemas.query import (
    QueryRequest,
    QueryResponse,
    SourceReference
)


router = APIRouter(tags=["Query"])


CITATION_PATTERN = re.compile(
    r"\[Source:\s*(.*?),\s*Page:\s*(\d+)\]"
)


@router.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest):

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    try:
        # 1. Retrieve relevant document chunks.
        retrieved_chunks = retrieve_documents(
            question,
            top_k=6
        )

        # 2. Generate an answer using the existing RAG generator.
        answer = generate_answer(
            question,
            retrieved_chunks
        )

        # 3. Extract sources actually cited in the answer.
        sources = []
        seen_sources = set()

        for match in CITATION_PATTERN.finditer(answer):

            source_path = match.group(1).strip()
            page = int(match.group(2))

            filename = Path(source_path).name
            source_key = (filename, page)

            if source_key not in seen_sources:
                sources.append(
                    SourceReference(
                        filename=filename,
                        page=page
                    )
                )
                seen_sources.add(source_key)

        # 4. Return the frontend-compatible response.
        return QueryResponse(
            question=question,
            answer=answer,
            sources=sources
        )

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your question."
        )