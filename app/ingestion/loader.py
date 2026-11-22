
import pymupdf
from pathlib import Path


def extract_page_text(page):

    width = page.rect.width
    height = page.rect.height

    if width / height > 1.2:

        midpoint = width / 2

        left_rect = pymupdf.Rect(
            0, 0, midpoint, height
        )

        right_rect = pymupdf.Rect(
            midpoint, 0, width, height
        )

        left_text = page.get_text(
            "text",
            clip=left_rect,
            sort=True
        ).strip()

        right_text = page.get_text(
            "text",
            clip=right_rect,
            sort=True
        ).strip()

        return "\n\n".join(
            part for part in [left_text, right_text] if part
        )

    return page.get_text("text", sort=True)


def load_pdf(file_path: str):

    documents = []

    pdf_path = Path(file_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported")

    with pymupdf.open(pdf_path) as pdf:

        for page_number, page in enumerate(pdf, start=1):

            text = extract_page_text(page)

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
