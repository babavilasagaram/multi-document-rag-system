
-- PostgreSQL schema for the Multi-Document RAG project
-- Requires the pgvector extension.

CREATE EXTENSION IF NOT EXISTS vector;

-- Sequences used by the existing database design
CREATE SEQUENCE IF NOT EXISTS documents_document_id_seq;
CREATE SEQUENCE IF NOT EXISTS document_chunks_id_seq;

-- Documents table
CREATE TABLE IF NOT EXISTS documents (
    document_id BIGINT NOT NULL
        DEFAULT nextval('documents_document_id_seq'::regclass),
    source TEXT NOT NULL,
    content_hash CHAR(64) NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT documents_pkey PRIMARY KEY (document_id)
);

ALTER SEQUENCE documents_document_id_seq
    OWNED BY documents.document_id;

CREATE UNIQUE INDEX IF NOT EXISTS documents_source_key
    ON documents (source);

-- Document chunks table
CREATE TABLE IF NOT EXISTS document_chunks (
    id BIGINT NOT NULL
        DEFAULT nextval('document_chunks_id_seq'::regclass),
    content TEXT NOT NULL,
    metadata JSONB NOT NULL,
    embedding VECTOR(384) NOT NULL,
    document_id BIGINT NOT NULL,
    chunk_index INTEGER NOT NULL,

    CONSTRAINT document_chunks_pkey PRIMARY KEY (id),

    CONSTRAINT fk_document_chunks_document
        FOREIGN KEY (document_id)
        REFERENCES documents(document_id)
        ON DELETE CASCADE
);

ALTER SEQUENCE document_chunks_id_seq
    OWNED BY document_chunks.id;

CREATE UNIQUE INDEX IF NOT EXISTS uq_document_chunk_index
    ON document_chunks (document_id, chunk_index);
