from pathlib import Path

from app.ingestion.pipeline import ingest_directory


PROJECT_ROOT = Path(__file__).resolve().parents[1]

data_directory = PROJECT_ROOT / "data"

results = ingest_directory(data_directory)

print("\nMULTI-DOCUMENT INGESTION SUMMARY")
print("=" * 50)

for result in results:
    print(
        f"{result['status']} | "
        f"{result['source']} | "
        f"Chunks: {result.get('chunks', 0)}"
    )

successful = sum(
    result["status"] in ("STORED", "SKIPPED")
    for result in results
)

failed = sum(
    result["status"] == "FAILED"
    for result in results
)

print(f"\nSuccessful: {successful}")
print(f"Failed: {failed}")