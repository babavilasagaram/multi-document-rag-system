import re
from pathlib import Path
import logging


from fastapi import APIRouter, HTTPException

from app.retrieval.retriever import retrieve_documents
from app.generation.generator import generate_answer

from app.api.schemas.query import (
    QueryRequest,
    QueryResponse,
    SourceReference
)

logger = logging.getLogger(__name__)
router = APIRouter(
    prefix="/query",
    tags=["Query"]
)


@router.post("", response_model=QueryResponse)
def query_documents(request: QueryRequest):

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    try:
        # 1. Retrieve relevant chunks.
        retrieved_chunks = retrieve_documents(
            question,
            top_k=6
        )

        # 2. Generate an answer using retrieved context.
        answer = generate_answer(
            question,
            retrieved_chunks
        )

        # 3. Extract citations from the formatted answer.
        citation_pattern = (
            r"\[Source:\s*(.*?),\s*Page:\s*(\d+)\]"
        )

        matches = re.findall(
            citation_pattern,
            answer
        )

        # 4. Build unique source references.
        sources = []
        seen = set()

        for source, page in matches:

            key = (source, int(page))

            if key not in seen:
                seen.add(key)

                sources.append(
                    SourceReference(
                        filename=Path(source).name,
                        page=int(page)
                    )
                )

        # 5. Return the API response.
        return QueryResponse(
            question=question,
            answer=answer,
            sources=sources
        )

    except Exception:
        logger.exception("Question processing failed.")

        raise HTTPException(
            status_code=500,
            detail="Question processing failed."
        )