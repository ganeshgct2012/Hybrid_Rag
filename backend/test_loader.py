import sys
from app.ingestion.loader import load_pdf, clean_documents

path = sys.argv[1]

raw = load_pdf(path)
clean = clean_documents(raw)

print(f"Pages loaded: {len(raw)}")
print(f"Pages after cleaning: {len(clean)}")
print("\n--- RAW page 1 (first 400 chars) ---")
print(raw[0].page_content[:400])
print("\n--- CLEAN page 1 (first 400 chars) ---")
print(clean[0].page_content[:400])
print("\nMetadata:", clean[0].metadata)