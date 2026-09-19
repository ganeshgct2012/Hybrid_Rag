from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    openai_api_key: str
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    # Neo4j Aura (cloud) - all three come from .env
    neo4j_uri: str
    neo4j_username: str = "neo4j"
    neo4j_password: str
    neo4j_database: str = "neo4j"

    chroma_dir: str = "./data/chroma"
    upload_dir: str = "./data/uploads"

    model_config = SettingsConfigDict(env_file="../.env", extra="ignore")

settings = Settings()