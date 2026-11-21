# Multi-Document RAG System

A Retrieval-Augmented Generation (RAG) application that processes multiple PDF documents, stores their embeddings in PostgreSQL using pgvector, retrieves relevant document chunks, and generates context-grounded answers with source citations.

## 1. Project Overview

This project explores the implementation and evaluation of a multi-document RAG pipeline.

The system supports:

- PDF text extraction and document chunking.
- Semantic embeddings using Hugging Face.
- Vector storage using PostgreSQL and pgvector.
- Document version identification using SHA-256 hashing.
- Idempotent document ingestion.
- Single-query and multi-query retrieval.
- Reciprocal Rank Fusion (RRF).
- LLM-based answer generation using Groq.
- Citation validation and source formatting.
- Retrieval and generation evaluation experiments.

## 2. Architecture

### Document Ingestion

PDF → Text Extraction → Chunking → Embeddings → PostgreSQL/pgvector

The ingestion pipeline:

1. Loads PDF documents.
2. Extracts page content and metadata.
3. Splits content using RecursiveCharacterTextSplitter.
4. Generates embeddings using `all-MiniLM-L6-v2`.
5. Stores document chunks and embeddings in PostgreSQL.

**Chunking configuration:**
- Chunk size: 800 characters
- Chunk overlap: 150 characters
- Splitting strategy: Recursive character splitting

### Document Versioning

The ingestion pipeline calculates a SHA-256 hash for each PDF.

This supports:
- Identifying unchanged documents.
- Skipping already-ingested document versions.
- Replacing chunks when a document changes.
- Maintaining document-level ingestion consistency.

### Retrieval

The system converts user questions into embeddings and searches PostgreSQL using pgvector's cosine-distance operator.

Two retrieval approaches are implemented:

**Single-query retrieval**

Retrieves the top-ranked document chunks for one query.

**Multi-query retrieval**

Processes multiple query variations independently and combines their rankings using Reciprocal Rank Fusion (RRF).

The RRF implementation uses a configurable constant, with a default value of 60.

### Answer Generation

The generation pipeline uses:
- Groq-hosted `openai/gpt-oss-120b`
- LangChain prompt templates
- LangChain output parsing

The model receives retrieved document context and generates an answer based on the provided material.

### Citation Validation

Generated answers use numbered source references.

The citation validator:
- Checks whether citations exist.
- Rejects invalid citation IDs.
- Converts valid references into source and page information.
- Preserves the configured abstention response when the answer is unavailable.

## 3. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| LangChain | Prompt and model pipeline |
| Hugging Face | Embedding generation |
| all-MiniLM-L6-v2 | Embedding model |
| Groq | LLM inference |
| PostgreSQL | Persistent storage |
| pgvector | Vector similarity search |
| PyMuPDF | PDF processing |
| pytest | Automated testing |

## 4. Project Structure

```text
RAG/
│
├── app/
│   ├── database/
│   │   ├── connection.py
│   │   ├── vector_store.py
│   │   └── test_db.py
│   │
│   ├── embeddings/
│   │   └── embedder.py
│   │
│   ├── generation/
│   │   └── generator.py
│   │
│   ├── ingestion/
│   │   ├── loader.py
│   │   ├── chunker.py
│   │   └── pipeline.py
│   │
│   └── retrieval/
│       └── retriever.py
│
├── data/
│   ├── evaluation_dataset.json
│   ├── evaluation_dataset_expansion_v1.json
│   ├── generation_evaluation_dataset.json
│   └── retrieval_comparison_results.json
│
├── evaluation/
│   ├── evaluate_retrieval.py
│   ├── evaluate_multi_query.py
│   ├── evaluate_multi_document_generation.py
│   ├── test_questions.json
│   └── ...
│
├── experiments/
│
├── tests/
│
├── .env.example
├── .gitignore
└── requirements.txt
```

## 5. Setup

### Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd RAG
```

### Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Configure environment variables

Create a `.env` file in the project root using `.env.example` as a reference.

Required variables:

```env
DATABASE_URL=postgresql://username:password@localhost:5432/database_name
GROQ_API_KEY=your_groq_api_key_here
```

Never commit your actual `.env` file or API keys.


### Database setup

This project requires PostgreSQL with the pgvector extension.

1. Create a PostgreSQL database.
2. Ensure pgvector is installed and available.
3. Execute the schema file:

```bash
psql -U postgres -d your_database -f database/schema.sql
```

4. Configure the database connection in the root `.env` file.

The schema creates:

- `documents`: stores document sources and SHA-256 hashes.
- `document_chunks`: stores extracted chunks, metadata and 384-dimensional embeddings.

A foreign key connects chunks to their parent documents. Deleting a document also deletes its associated chunks.

## 6. Running the Pipeline

The ingestion pipeline exposes:

```python
from app.ingestion.pipeline import ingest_pdf, ingest_directory

result = ingest_pdf("data/example.pdf")
print(result)
```

For directory ingestion:

```python
results = ingest_directory("data")
print(results)
```

Unchanged documents can be skipped based on their content hash.

## 7. Retrieval

Single-query retrieval:

```python
from app.retrieval.retriever import retrieve_documents

results = retrieve_documents(
    "What is the Transformer architecture?",
    top_k=5
)
```

Multi-query retrieval:

```python
from app.retrieval.retriever import retrieve_multi_query

queries = [
    "What is positional encoding?",
    "Why does the Transformer need positional information?"
]

results = retrieve_multi_query(
    queries,
    top_k_per_query=5,
    final_k=10
)
```

## 8. Testing

Run automated tests:

```bash
python -m pytest -v
```

The current verified automated test baseline contains six tests covering:

- Chunk generation.
- Metadata preservation.
- Chunk-size constraints.
- Citation formatting.
- Invalid citation rejection.
- Missing citation rejection.

These tests do not represent complete end-to-end RAG validation.

## 9. Evaluation and Experiments

The repository contains evaluation datasets and scripts for investigating:

- Retrieval performance.
- Multi-query retrieval.
- RRF ranking.
- Prompt comparisons.
- Generation quality.
- Citation behavior.

Performance claims should be based on the measured evaluation results, not simply on the presence of an implemented technique.

## 10. Current Limitations

- Automated unit-test coverage is currently limited.
- Full clean-environment installation has not yet been verified.
- Database-dependent integration tests require a configured PostgreSQL instance.
- Production-scale performance and reliability have not been established.
- Retrieval and generation quality should be assessed using the supplied evaluation methodology.

## 11. Project Goal

The goal is to understand, implement, evaluate, and progressively improve a practical multi-document RAG system, with particular attention to retrieval quality, answer grounding, citation correctness, and reproducibility.