import sys
from app.retrieval.hybrid import answer_question

question = " ".join(sys.argv[1:]) or "What technologies does the frontend use?"

result = answer_question(question)

print("QUESTION:", question)
print("\nGRAPH ENTITIES MATCHED:", result["entities"])

print(f"\nGRAPH FACTS USED ({len(result['graph_facts'])}):")
for fact in result["graph_facts"][:15]:
    print("  ", fact)

print("\nVECTOR PAGES USED:", result["pages"])

print("\nANSWER:")
print(result["answer"])