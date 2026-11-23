from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import documents, query

from app.api.routes import documents


app = FastAPI(
    title="Multi-Document RAG API",
    description="API for querying and managing documents using RAG.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8443",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "message": "RAG API is running"
    }


app.include_router(documents.router)
app.include_router(documents.router)
app.include_router(query.router)
