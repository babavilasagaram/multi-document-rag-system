from app.ingestion.pipeline import ingest_pdf

result = ingest_pdf(
    "data/NIPS-2017-attention-is-all-you-need-Paper.pdf"
)

print("\nIngestion Result:")
print(result)