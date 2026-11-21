
import pymupdf
from pathlib import Path


def load_pdf(file_path: str):

    documents = []

    pdf_path = Path(file_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported")

    with pymupdf.open(pdf_path) as pdf:

        for page_number, page in enumerate(pdf, start=1):

            text = page.get_text("text")

            if text.strip():

                documents.append({
                    "text": text,
                    "metadata": {
                        "source": pdf_path.name,
                        "page": page_number
                    }
                })

    if not documents:
        raise ValueError(
            "No extractable text found. "
            "The PDF may contain scanned images."
        )

    return documents
