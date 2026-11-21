from langchain_huggingface import HuggingFaceEmbeddings


model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)


def embed_documents(chunks):

    texts = [chunk["text"] for chunk in chunks]

    vectors = model.embed_documents(texts)

    embedded_chunks = []

    for chunk, vector in zip(chunks, vectors):

        embedded_chunks.append({
            "text": chunk["text"],
            "metadata": chunk["metadata"],
            "embedding": vector
        })

    return embedded_chunks


def embed_query(query):

    return model.embed_query(query)