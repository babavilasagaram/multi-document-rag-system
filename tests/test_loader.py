
from app.ingestion.loader import load_pdf

documents = load_pdf(
    "data/NIPS-2017-attention-is-all-you-need-Paper.pdf"
)

print("Total extracted pages:", len(documents))

for document in documents[:2]:

    print("\nMetadata:", document["metadata"])

    print("Extracted text:")
    print(document["text"][:500])
