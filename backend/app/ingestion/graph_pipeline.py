import re
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from app.config import settings

# Manual aliases: variant spelling (lowercase, letters/digits only) -> canonical name
ALIASES = {
    "nextjs": "Next.js",
    "reactquery": "TanStack Query",
    "tanstackquery": "TanStack Query",
    "tailwind": "Tailwind CSS",
    "tailwindcss": "Tailwind CSS",
    "hlsjs": "HLS.js",
    "oauth2": "OAuth 2.0",
    "oauth": "OAuth 2.0",
    "pwa": "PWA",
    "nestjs": "NestJS",
    "nodejs": "Node.js",
    "postgresql": "PostgreSQL",
    "postgres": "PostgreSQL",
    "clickhouse": "ClickHouse",
    "apigateway": "API Gateway",
    "rest": "REST",
    "graphql": "GraphQL",
    "rabbitmq": "RabbitMQ",
    "oauth2oidc": "OAuth 2.0",
}


def get_graph() -> Neo4jGraph:
    return Neo4jGraph(
        url=settings.neo4j_uri,
        username=settings.neo4j_username,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
    )


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        temperature=0,
    )


def chunk_for_graph(docs: list[Document]) -> list[Document]:
    """Bigger chunks than the vector pipeline, so the LLM sees more context per call."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    return splitter.split_documents(docs)


def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def normalize_graph_documents(graph_docs):
    """Make 'Next.js', 'nextjs' and 'NEXT JS' the same node."""
    canonical = {}  # key -> chosen spelling (first one seen wins)

    def canon(name: str) -> str:
        name = re.sub(r"\s+", " ", str(name)).strip()
        k = _key(name)
        if k in ALIASES:
            return ALIASES[k]
        return canonical.setdefault(k, name)

    for gd in graph_docs:
        for node in gd.nodes:
            node.id = canon(node.id)
            node.type = str(node.type).strip().title().replace(" ", "")
        for rel in gd.relationships:
            rel.source.id = canon(rel.source.id)
            rel.source.type = str(rel.source.type).strip().title().replace(" ", "")
            rel.target.id = canon(rel.target.id)
            rel.target.type = str(rel.target.type).strip().title().replace(" ", "")
            rel.type = re.sub(r"\W+", "_", str(rel.type).strip()).upper()
    return graph_docs


def ingest_to_graph(docs: list[Document], doc_id: str) -> dict:
    """Extract entities/relations, normalize, and store in Neo4j."""
    chunks = chunk_for_graph(docs)
    for c in chunks:
        c.metadata["doc_id"] = doc_id

    transformer = LLMGraphTransformer(
        llm=get_llm(),
        allowed_nodes=["Layer", "Technology", "Service", "Database", "Feature", "Protocol"],
        allowed_relationships=["USES", "PART_OF", "DEPENDS_ON", "STORES_DATA_IN", "PROVIDES"],
    )
    graph_docs = transformer.convert_to_graph_documents(chunks)
    graph_docs = normalize_graph_documents(graph_docs)

    graph = get_graph()
    graph.add_graph_documents(graph_docs, baseEntityLabel=True, include_source=True)

    return {
        "chunks": len(chunks),
        "nodes": sum(len(g.nodes) for g in graph_docs),
        "relationships": sum(len(g.relationships) for g in graph_docs),
    }