from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from app.config import settings

COLLECTION = "pdf_chunks"


def chunk_documents(docs: list[Document]) -> list[Document]:
    """Split pages into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def get_vector_store() -> Chroma:
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=settings.chroma_dir,
    )


def ingest_to_vector_db(docs: list[Document], doc_id: str) -> int:
    """Chunk, embed and store. Returns the number of chunks stored."""
    chunks = chunk_documents(docs)
    for i, chunk in enumerate(chunks):
        chunk.metadata["doc_id"] = doc_id
        chunk.metadata["chunk_id"] = f"{doc_id}-{i}"
    store = get_vector_store()
    store.add_documents(chunks, ids=[c.metadata["chunk_id"] for c in chunks])
    return len(chunks)