import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def load_pdf(file_path: str) -> list[Document]:
    """Load a PDF. Returns one Document per page, with metadata (source, page)."""
    loader = PyPDFLoader(file_path)
    return loader.load()


def clean_text(text: str) -> str:
    """Clean messy PDF text but keep headings and bullet structure."""
    text = text.replace("\x00", "")
    text = re.sub(r"^\s*[=\-_*~]{3,}\s*$", "", text, flags=re.MULTILINE)  # separator lines (=====, -----)
    text = re.sub(r"(\w)-\n([a-z])", r"\1\2", text)      # join words hyphen-split across lines
    text = re.sub(r"(?<=[a-z,])\n(?=[a-z])", " ", text)  # join lines wrapped mid-sentence only
    text = re.sub(r"[ \t]+", " ", text)                  # collapse repeated spaces
    text = re.sub(r"\n{3,}", "\n\n", text)               # max one blank line
    return text.strip()


def clean_documents(docs: list[Document]) -> list[Document]:
    """Clean every page and drop pages that end up empty."""
    cleaned = []
    for doc in docs:
        text = clean_text(doc.page_content)
        if text:
            cleaned.append(Document(page_content=text, metadata=doc.metadata))
    return cleaned