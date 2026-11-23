from fastapi import FastAPI

from app.api.routes import documents, query


app = FastAPI(
    title="Multi-Document RAG API",
    description="API for querying and managing documents using RAG.",
    version="1.0.0"
)


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "message": "RAG API is running"
    }


app.include_router(documents.router)
app.include_router(query.router)