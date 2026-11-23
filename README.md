# Multi-Document RAG System

A full-stack, multi-document Retrieval-Augmented Generation (RAG) application built using **Python, FastAPI, LangChain, PostgreSQL, pgvector, Groq, React and TypeScript**.

The application allows users to upload PDF documents, retrieve relevant information from multiple documents, and generate context-grounded answers with source references through a web interface.

The project explores the complete RAG workflow: PDF processing, document chunking, embedding generation, vector storage, semantic retrieval, query expansion, Reciprocal Rank Fusion (RRF), grounded answer generation and citation validation.

Rather than focusing only on building a working RAG pipeline, this project also investigates how retrieval strategies and generation prompts affect answer quality.

---

## 1. Project Overview

The system retrieves information from multiple PDF documents and generates answers grounded in the retrieved content.

It consists of two main components:

- **Backend:** Handles document ingestion, embeddings, vector storage, retrieval, LLM generation and API requests.
- **Frontend:** Provides a web interface for uploading documents, viewing available documents and asking questions.

### Key Features

**Backend and RAG**

- Multi-document PDF ingestion.
- PDF text extraction using PyMuPDF.
- Wide-page PDF handling using separate left/right extraction.
- Recursive text chunking with configurable overlap.
- Semantic embeddings using Sentence Transformers.
- PostgreSQL vector storage with pgvector.
- SHA-256-based document version identification.
- Extraction-version-aware re-ingestion.
- Idempotent document ingestion.
- Single-query semantic retrieval.
- Multi-query retrieval.
- Reciprocal Rank Fusion (RRF).
- LLM-based answer generation using Groq.
- Numbered source citations with page references.
- Citation reference validation.
- Retrieval and generation evaluation experiments.

**Frontend**

- React-based user interface.
- PDF document upload.
- Upload progress tracking.
- Display of available documents.
- Question-and-answer interface.
- Display of generated answers and source references.
- Backend health-check integration.
- API error handling.

---

## 2. System Architecture

The application consists of three main workflows.

### A. Document Ingestion

PDF Documents → Text Extraction → Chunking → Embedding Generation → PostgreSQL / pgvector

The ingestion pipeline:

1. Loads PDF documents using PyMuPDF.
2. Extracts text page by page while preserving source and page metadata.
3. Detects wide pages and extracts their left and right halves separately.
4. Splits extracted text into smaller chunks.
5. Generates 384-dimensional embeddings.
6. Stores document information, chunks and embeddings in PostgreSQL.

**Chunking configuration**

| Parameter            | Value                          |
| -------------------- | ------------------------------ |
| Chunk size           | 800 characters                 |
| Chunk overlap        | 150 characters                 |
| Chunking method      | RecursiveCharacterTextSplitter |
| Embedding model      | all-MiniLM-L6-v2               |
| Embedding dimensions | 384                            |

### B. Question Answering

User Question → Query Embedding → Vector Retrieval → Context Preparation → LLM Generation → Citation Validation → Final Answer

The question-answering pipeline:

1. Converts the user's question into an embedding.
2. Retrieves relevant document chunks using vector similarity.
3. Optionally retrieves results from multiple query variations.
4. Combines rankings using Reciprocal Rank Fusion.
5. Passes retrieved context to the language model.
6. Generates an answer using the supplied document context.
7. Validates citation references and formats source information.

### C. Frontend and Backend Communication

The React frontend communicates with the FastAPI backend through HTTP API requests.

The frontend supports the following operations:

| Operation                    | API endpoint        | Method |
| ---------------------------- | ------------------- | ------ |
| Backend health check         | `/health`           | GET    |
| Retrieve available documents | `/documents`        | GET    |
| Upload a PDF document        | `/documents/upload` | POST   |
| Ask a question               | `/query`            | POST   |

The frontend API client uses `VITE_API_BASE_URL` to configure the backend address.

---

## 3. Document Ingestion and Versioning

The ingestion pipeline uses SHA-256 hashing to identify document versions.

It combines the original PDF hash with an extraction version identifier.

- If the PDF and extraction version are unchanged, ingestion skips the document.
- If the PDF changes, the document is reprocessed.
- If the extraction logic version changes, the document is also reprocessed.
- Existing chunks are replaced when a new version is ingested.

This helps prevent unnecessary embedding generation and keeps stored chunks consistent with the extraction logic.

**Current extraction version:** `v2`

The current PDF extraction approach uses PyMuPDF. Wide pages are split into left and right extraction regions to improve reading order in two-page spreads.

---

## 4. Retrieval Strategies

The system implements two retrieval approaches.

### Single-Query Retrieval

The user's question is converted into an embedding, and PostgreSQL retrieves the most similar document chunks.

The vector search uses pgvector's cosine-distance operator:

`<=>`

Similarity is calculated as:

`Similarity = 1 - Cosine Distance`

### Multi-Query Retrieval

Instead of relying on a single formulation of a question, the system accepts multiple query variations.

Each query independently retrieves relevant chunks. Their results are then combined using Reciprocal Rank Fusion.

### Reciprocal Rank Fusion (RRF)

RRF combines rankings from multiple retrieval results without requiring their similarity scores to be directly comparable.

[
RRF(d)=\sum\_{r \in R}\frac{1}{k+rank_r(d)}
]

Where:

- (d) is a document chunk.
- (R) represents the retrieval rankings.
- (rank_r(d)) is the position of the chunk in a ranking.
- (k) is a configurable constant, defaulting to 60.

The final results are ranked using their fused scores.

---

## 5. Answer Generation and Citation Validation

The generation component uses:

- Groq-hosted `openai/gpt-oss-120b`.
- LangChain prompt templates.
- Context-grounded answer generation.
- Configurable prompt templates.
- Explicit abstention instructions.

The model receives retrieved document chunks with numbered source references.

The prompt instructs the model to:

- Answer using the provided context.
- Avoid inventing facts or references.
- Abstain when the context does not contain the answer.
- Use the supplied source references.

### Citation Validation

The citation validator checks generated references before formatting them.

It supports citation formats such as:

- `[1]`
- `【1】`
- `【1†L1-L4】`

Valid references are converted into source and page information.

**Important limitation:** Citation validation checks reference presence and validity. It does not independently prove that a cited passage semantically supports the generated claim. Invalid or missing citations produce an error response; automatic regeneration is not implemented.

---

## 6. Technology Stack

### Backend

| Technology            | Purpose                            |
| --------------------- | ---------------------------------- |
| Python                | Main programming language          |
| FastAPI               | Backend API                        |
| LangChain             | LLM prompt and generation pipeline |
| Sentence Transformers | Embedding generation               |
| all-MiniLM-L6-v2      | Embedding model                    |
| Groq                  | LLM inference                      |
| PostgreSQL            | Persistent storage                 |
| pgvector              | Vector similarity search           |
| PyMuPDF               | PDF text extraction                |
| pytest                | Automated testing                  |

### Frontend

| Technology     | Purpose                                |
| -------------- | -------------------------------------- |
| React 19       | User interface                         |
| TypeScript     | Type-safe frontend development         |
| Vite 8         | Frontend development and build tooling |
| Tailwind CSS 4 | Styling                                |
| lucide-react   | Icons                                  |
| pnpm           | Package management                     |

---

## 7. Project Structure

```text
multi-document-rag-system/
│
├── app/                       # Backend application
│   ├── api/                   # FastAPI application and routes
│   ├── database/              # Database connection and vector store
│   ├── embeddings/            # Embedding generation
│   ├── generation/            # LLM answer generation
│   ├── ingestion/             # PDF loading and chunking
│   └── retrieval/             # Semantic and multi-query retrieval
│
├── database/                  # Database schema
│
├── data/                      # Evaluation data and results
│
├── evaluation/                # RAG evaluation scripts and results
│
├── experiments/               # Experimental workflows
│
├── scripts/                   # Utility scripts
│
├── tests/                     # Automated and experimental tests
│
├── src/                       # React frontend source
├── public/                    # Static frontend assets
│
├── index.html                 # Frontend HTML entry point
├── package.json               # Frontend dependencies and scripts
├── pnpm-lock.yaml             # Frontend dependency lockfile
├── vite.config.ts             # Vite configuration
├── tsconfig.json              # TypeScript configuration
│
├── .env.example               # Environment variable template
├── .gitignore
├── requirements.txt           # Backend dependencies
└── README.md
```

---

## 8. Installation and Configuration

### Prerequisites

- Python 3.11 or compatible version.
- Node.js compatible with the installed Vite version.
- pnpm.
- PostgreSQL.
- pgvector extension.
- Groq API key.

### Clone the Repository

```bash
git clone https://github.com/babavilasagaram/multi-document-rag-system.git

cd multi-document-rag-system
```

### A. Backend Setup

**Create a virtual environment**

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

**Install backend dependencies**

```bash
pip install -r requirements.txt
```

**Configure environment variables**

Create a `.env` file in the project root using `.env.example` as a reference.

```env
DATABASE_URL=postgresql://username:password@localhost:5432/rag_db

GROQ_API_KEY=your_groq_api_key
```

Never commit your actual `.env` file or API keys.

### B. Database Setup

The project requires PostgreSQL with the pgvector extension.

1. Create a PostgreSQL database.
2. Ensure pgvector is installed.
3. Execute the database schema:

```bash
psql -U postgres -d rag_db -f database/schema.sql
```

The schema defines:

- `documents`: document source information and content hashes.
- `document_chunks`: extracted text, metadata and 384-dimensional embeddings.

Document chunks are linked to their parent documents using a foreign key.

### C. Frontend Setup

The frontend is located in the repository root.

**Install frontend dependencies**

```bash
pnpm install
```

**Configure the backend API URL**

The frontend reads the backend address from `VITE_API_BASE_URL`.

For local development, the frontend defaults to:

```text
http://127.0.0.1:8000
```

To override it, create a `.env.local` file in the project root:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

For deployment, replace this value with the deployed backend URL.

---

## 9. Supplying PDF Documents

The original PDF documents used during development are intentionally not included in this GitHub repository.

Users must provide their own PDF documents before running the ingestion pipeline.

1. Create the `data/` directory if it does not already exist.
2. Place your PDF documents inside it.
3. Run the ingestion pipeline on the supplied documents.

Example:

```text
data/
├── example_document_1.pdf
├── example_document_2.pdf
└── example_document_3.pdf
```

The project does not automatically download or provide the original development PDFs.

**Important:** Do not commit private, copyrighted or sensitive documents to your own public repository unless you have permission to share them. Keep actual source documents out of Git unless redistribution is permitted.

---

## 10. Running the Application

The backend and frontend run as separate development processes.

### A. Start the Backend

Activate the Python virtual environment and run:

```bash
uvicorn app.api.main:app --reload
```

The backend should be available at:

```text
http://127.0.0.1:8000
```

FastAPI documentation:

```text
http://127.0.0.1:8000/docs
```

### B. Start the Frontend

Open a separate terminal in the project root:

```bash
pnpm dev
```

Vite will display the local development URL, typically:

```text
http://localhost:5173
```

Open that URL in your browser.

### C. Build the Frontend

To generate a production frontend build:

```bash
pnpm build
```

To preview the production build locally:

```bash
pnpm preview
```

---

## 11. Running the RAG Pipeline Directly

### Ingest a Single PDF

```python
from app.ingestion.pipeline import ingest_pdf

result = ingest_pdf("data/example_document.pdf")
print(result)
```

### Ingest a Directory

```python
from app.ingestion.pipeline import ingest_directory

results = ingest_directory("data")
print(results)
```

Unchanged document versions are skipped automatically.

### Run Retrieval

```python
from app.retrieval.retriever import retrieve_documents

results = retrieve_documents(
    "What is the Transformer architecture?",
    top_k=5
)
```

### Run Multi-Query Retrieval

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

---

## 12. Evaluation and Experimental Results

An important part of this project is evaluating retrieval and generation rather than assuming that a working pipeline produces reliable answers.

### A. Retrieval Evaluation

A manually judged dataset containing 20 questions about the original Transformer paper was used to compare retrieval approaches.

**Historical retrieval results**

| Strategy          |  Recall\@5 | Precision\@5 |     MRR\@5 |    nDCG\@5 |
| ----------------- | ---------: | -----------: | ---------: | ---------: |
| Baseline          |     60.39% |       55.00% |     90.00% |     68.77% |
| Candidate Pooling |     64.27% |       58.00% | **94.17%** |     72.35% |
| RRF               | **65.11%** |   **59.00%** |     91.00% | **73.76%** |

**Observations**

- RRF achieved the highest Recall\@5, Precision\@5 and nDCG\@5.
- Candidate Pooling achieved the highest MRR\@5.
- Multi-query approaches improved retrieval coverage over the baseline in this experiment.

**Evaluation limitations**

- The dataset contains only 20 questions.
- All questions focus on one research paper.
- Relevance judgments were manually created using a pooled candidate set and are not exhaustive.
- The comparison does not isolate the effect of fusion from query expansion.

These results represent a historical experiment, not current retrieval performance after PDF extraction v2. Re-ingestion generated new chunk IDs, so the earlier relevance judgments cannot be used directly against the updated database without remapping and re-evaluation.

### B. Generation Evaluation

A manually reviewed evaluation set contained:

- 16 questions.
- 12 answerable questions.
- 4 unanswerable questions.

**Historical manual quality review**

| Metric                 |            Score |
| ---------------------- | ---------------: |
| Overall quality        | 123/128 (96.09%) |
| Answerable questions   |            91/96 |
| Unanswerable questions |            32/32 |

The review considered:

- Factual correctness.
- Semantic faithfulness.
- Completeness.
- Citation accuracy.

The unanswerable questions were correctly handled in all four reviewed cases.

**Important:** These are manually assigned scores on a small curated evaluation set. They are not automated measures of general answer accuracy or proof of production reliability. The generation evaluation also predates the PDF extraction v2 update.

The original generation evaluation dataset is not included in the tracked repository files. The reported historical results are retained for project documentation, but the experiment is not fully reproducible from the repository alone.

### C. Prompt Experiments

Additional experiments investigated:

- Full prompt comparisons.
- Completeness-focused instructions.
- Context ordering.
- Context additions.
- Generalization of prompt changes.

The experiments showed that adding more instructions does not necessarily improve answer quality. Some prompt changes improved individual responses, while others reduced the aggregate manual quality score.

No experimental prompt modification was adopted into the production generator solely on the basis of an isolated improvement.

---

## 13. Testing

Run the automated test suite:

```bash
python -m pytest -v
```

The current verified baseline contains six automated tests covering:

- Chunk generation.
- Metadata preservation.
- Chunk-size constraints.
- Citation formatting.
- Invalid citation rejection.
- Missing citation rejection.

Other scripts under `tests/` are used for manual checks, integration workflows, retrieval experiments and generation experiments.

The six-test baseline does not represent comprehensive end-to-end validation.

---

## 14. Current Limitations

- Scanned PDFs are not supported through OCR.
- Complex PDF layouts, tables, charts and diagrams may not be extracted accurately.
- Wide-page extraction improves reading order for tested two-page spreads but is not a universal layout parser.
- Citation validation does not establish semantic entailment between claims and cited passages.
- The retrieval evaluation dataset is small and focused on one research paper.
- Automated test coverage is limited.
- Full clean-environment installation has not yet been verified.
- Production-scale performance, concurrency and reliability have not been established.
- The application has not yet been validated under production deployment conditions.

---

## 15. Project Objective

The objective is to understand and implement a practical full-stack multi-document RAG system while developing the ability to evaluate its individual components.

The project emphasizes:

- Reliable document ingestion.
- Retrieval quality and ranking.
- Context-grounded answer generation.
- Citation correctness.
- Frontend and backend integration.
- Reproducible experiments.
- Evidence-based engineering decisions.

The focus is not simply to make a RAG application work, but to understand where it succeeds, where it fails and how its performance can be measured and improved.
