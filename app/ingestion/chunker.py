
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""]
    )

    texts = [doc["text"] for doc in documents]
    metadatas = [doc["metadata"] for doc in documents]

    chunks = splitter.create_documents(
        texts,
        metadatas=metadatas
    )

    return [
        {
            "text": chunk.page_content,
            "metadata": chunk.metadata
        }
        for chunk in chunks
    ]
