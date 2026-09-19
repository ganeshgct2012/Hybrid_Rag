import sys
from app.ingestion.loader import load_pdf
from app.ingestion.graph_pipeline import ingest_to_graph, get_graph

path = sys.argv[1]

docs = load_pdf(path)
stats = ingest_to_graph(docs, doc_id="sample")
print("Ingested:", stats)

graph = get_graph()
print("\nEntity count:", graph.query("MATCH (n:__Entity__) RETURN count(n) AS c"))
print("\nSample relationships:")
rows = graph.query(
    "MATCH (a:__Entity__)-[r]->(b:__Entity__) "
    "WHERE type(r) <> 'MENTIONS' "
    "RETURN a.id AS source, type(r) AS relation, b.id AS target LIMIT 15"
)
for r in rows:
    print(f"  {r['source']}  -[{r['relation']}]->  {r['target']}")