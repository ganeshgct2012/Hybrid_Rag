from langchain_neo4j import Neo4jGraph
from app.config import settings

graph = Neo4jGraph(
    url=settings.neo4j_uri,
    username=settings.neo4j_username,
    password=settings.neo4j_password,
    database=settings.neo4j_database,
)
print(graph.query("RETURN 'Aura connected' AS msg"))