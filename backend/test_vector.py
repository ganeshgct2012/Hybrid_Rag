import sys
from app.ingestion.loader import load_pdf, clean_documents
from app.ingestion.vector_pipeline import ingest_to_vector_db, get_vector_store

path = sys.argv[1]

docs = clean_documents(load_pdf(path))
n = ingest_to_vector_db(docs, doc_id="sample")
print(f"Stored {n} chunks in Chroma")

store = get_vector_store()
results = store.similarity_search("Which frontend framework is used?", k=3)
for r in results:
    print("\n--- page", r.metadata.get("page"), "---")
    print(r.page_content[:300])