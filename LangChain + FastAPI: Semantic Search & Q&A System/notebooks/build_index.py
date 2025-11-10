import os
import logging
from pathlib import Path
from typing import List

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    Docx2txtLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Optional: force transformers to *not* use TensorFlow / Flax / JAX
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["TRANSFORMERS_NO_JAX"] = "1"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("build_index")


class Settings:
    """Simple config holder for paths and hyperparameters."""

    def __init__(
        self,
        data_dir: str = "docs",
        index_dir: str = "index",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
    ):
        self.data_dir = os.getenv("DATA_DIR", data_dir)
        self.index_dir = os.getenv("INDEX_DIR", index_dir)
        self.model_name = os.getenv("MODEL_NAME", model_name)
        self.chunk_size = int(os.getenv("CHUNK_SIZE", chunk_size))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", chunk_overlap))


settings = Settings()


def load_documents(data_dir: str) -> List[Document]:
    """
    Load .txt and .docx documents from a directory into LangChain Documents.
    Each document gets a 'source' metadata field (filename).
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_path.resolve()}")

    docs: List[Document] = []

    # .txt files
    txt_loader = DirectoryLoader(
        str(data_path),
        glob="**/*.txt",
        loader_cls=TextLoader,
        show_progress=True,
    )

    # .docx files (if any)
    docx_loader = DirectoryLoader(
        str(data_path),
        glob="**/*.docx",
        loader_cls=Docx2txtLoader,
        show_progress=True,
    )

    for loader in [txt_loader, docx_loader]:
        loaded_docs = loader.load()
        for doc in loaded_docs:
            raw_source = doc.metadata.get("source", "unknown")
            doc.metadata["source"] = Path(raw_source).name
            docs.append(doc)

    logger.info("Loaded %d documents from %s", len(docs), data_path)
    return docs


def build_faiss_index(
    data_dir: str = settings.data_dir,
    index_dir: str = settings.index_dir,
    model_name: str = settings.model_name,
    chunk_size: int = settings.chunk_size,
    chunk_overlap: int = settings.chunk_overlap,
) -> None:
    """
    Build a FAISS index from documents in data_dir and save it to index_dir.
    """
    logger.info("🔧 Starting FAISS index build...")
    docs = load_documents(data_dir)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(docs)
    logger.info("📚 Split into %d chunks", len(chunks))

    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vectordb = FAISS.from_documents(chunks, embeddings)

    index_path = Path(index_dir)
    index_path.mkdir(parents=True, exist_ok=True)
    vectordb.save_local(str(index_path))

    logger.info("✅ Index saved to %s", index_path.resolve())


if __name__ == "__main__":
    build_faiss_index()
