import difflib
import re
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from app.ingestion.vector_pipeline import get_vector_store
from app.ingestion.graph_pipeline import get_graph, get_llm


# ---------- Part 1: find which graph entities the question is about ----------

class QuestionEntities(BaseModel):
    names: list[str] = Field(
        description="Technologies, services, databases, components or layers mentioned in the question"
    )


_entity_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Extract the entity names (technologies, services, databases, components, layers) "
     "mentioned in the question. Return only names that appear in the question. "
     "If there are none, return an empty list."),
    ("human", "{question}"),
])


def extract_entities_with_llm(question: str) -> list[str]:
    try:
        chain = _entity_prompt | get_llm().with_structured_output(QuestionEntities)
        return chain.invoke({"question": question}).names
    except Exception:
        return []


def _key(text: str) -> str:
    """Lowercase and keep only letters/digits: 'Front End' and 'Frontend' both become 'frontend'."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def find_entities_in_graph(graph, question: str) -> list[str]:
    """Check every entity name stored in the graph against the question text.

    1. Word groups of 1-4 words are joined without spaces, so 'front end' matches
       the 'Frontend' node and 'tan stack query' matches 'TanStack Query'.
    2. Close spellings are matched too, so the typo 'frond end' still finds 'Frontend'.

    This does not depend on the LLM."""
    rows = graph.query("MATCH (e:__Entity__) RETURN DISTINCT e.id AS id")

    # normalized key -> original entity name
    entity_keys = {}
    for r in rows:
        name = str(r["id"])
        k = _key(name)
        if len(k) >= 3:
            entity_keys.setdefault(k, name)

    words = re.findall(r"[A-Za-z0-9]+", question)
    phrases = set()
    for size in (1, 2, 3, 4):
        for i in range(len(words) - size + 1):
            phrases.add(_key("".join(words[i:i + size])))

    # 1. exact match after normalizing
    found = [name for k, name in entity_keys.items() if k in phrases]

    # 2. typo tolerance (only for longer names, to avoid false matches)
    long_keys = [k for k in entity_keys if len(k) >= 7]
    for phrase in phrases:
        if len(phrase) < 7:
            continue
        for k in difflib.get_close_matches(phrase, long_keys, n=1, cutoff=0.85):
            name = entity_keys[k]
            if name not in found:
                found.append(name)

    return found


# ---------- Part 2: graph search (Neo4j) ----------

GRAPH_QUERY = """
MATCH (e:__Entity__)
WHERE toLower(e.id) = toLower($name)
   OR ($fuzzy AND toLower(e.id) CONTAINS toLower($name))
MATCH (e)-[r]-(n:__Entity__)
RETURN e.id AS entity,
       type(r) AS relation,
       n.id AS other,
       startNode(r) = e AS outgoing
LIMIT $limit
"""


def graph_search(question: str, limit: int = 25) -> tuple[list[str], list[str]]:
    """Returns (facts, entities_used). Facts are 1-hop relationships as text."""
    graph = get_graph()

    exact = find_entities_in_graph(graph, question)
    fuzzy = [n for n in extract_entities_with_llm(question) if len(n) >= 3 and n not in exact]

    facts = []
    used = list(exact)
    for name, is_fuzzy in [(n, False) for n in exact] + [(n, True) for n in fuzzy]:
        rows = graph.query(
            GRAPH_QUERY, params={"name": name, "fuzzy": is_fuzzy, "limit": limit}
        )
        if is_fuzzy and rows:
            used.append(name)
        for r in rows:
            if r["outgoing"]:
                facts.append(f"{r['entity']} -[{r['relation']}]-> {r['other']}")
            else:
                facts.append(f"{r['other']} -[{r['relation']}]-> {r['entity']}")

    return list(dict.fromkeys(facts)), used


# ---------- Part 3: vector search (Chroma) ----------

def vector_search(question: str, k: int = 4):
    return get_vector_store().similarity_search(question, k=k)


# ---------- Part 4: combine both and answer ----------

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You answer questions about a document using two sources: "
     "(1) document excerpts found by vector search, and "
     "(2) knowledge-graph facts showing how entities relate to each other. "
     "Use only this context. If the answer is not in the context, say you don't know. "
     "Mention page numbers when the excerpts give them."),
    ("human",
     "Document excerpts:\n{chunks}\n\n"
     "Knowledge graph facts:\n{facts}\n\n"
     "Question: {question}"),
])


def answer_question(question: str) -> dict:
    chunks = vector_search(question)
    facts, entities = graph_search(question)

    chunk_text = "\n\n".join(
        f"[page {c.metadata.get('page_label', '?')}] {c.page_content}" for c in chunks
    )
    fact_text = "\n".join(facts) if facts else "(no graph facts found for this question)"

    response = (ANSWER_PROMPT | get_llm()).invoke(
        {"chunks": chunk_text, "facts": fact_text, "question": question}
    )

    return {
        "answer": response.content,
        "entities": entities,
        "graph_facts": facts,
        "pages": sorted({c.metadata.get("page_label", "?") for c in chunks}),
    }