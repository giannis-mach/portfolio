import os
import time
import logging
from pathlib import Path
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Avoid TensorFlow / Flax / JAX in transformers
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["TRANSFORMERS_NO_JAX"] = "1"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("langchain_fastapi_rag")


class Settings(BaseModel):
    data_dir: str = "docs"
    index_dir: str = "index"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k_default: int = 4


settings = Settings()


def load_vectorstore(
    index_dir: str = settings.index_dir,
    model_name: str = settings.model_name,
) -> FAISS:
    """
    Load FAISS index from disk with the same embedding model.
    """
    index_path = Path(index_dir)
    if not index_path.exists():
        raise FileNotFoundError(
            f"Index directory does not exist: {index_path.resolve()}. "
            f"Run `python build_index.py` first."
        )

    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vectordb = FAISS.load_local(
        str(index_path),
        embeddings,
        allow_dangerous_deserialization=True,  # OK for local dev only
    )
    logger.info("📦 Loaded FAISS index from %s", index_path.resolve())
    return vectordb


def simple_answer(question: str, docs: List[Document]) -> str:
    """
    Very simple local 'answer generator':
    - Tokenizes question into words
    - Splits each document into sentences
    - Scores sentences by word overlap with question
    - Returns the best-matching sentence as the answer
    """
    import re

    if not docs:
        return "I couldn't find relevant information in the indexed documents."

    q_words = set(re.findall(r"\w+", question.lower()))

    best_sentence = ""
    best_score = 0

    for doc in docs:
        sentences = re.split(r"[.!?]\s+", doc.page_content)
        for sent in sentences:
            s_words = set(re.findall(r"\w+", sent.lower()))
            score = len(q_words & s_words)
            if score > best_score:
                best_score = score
                best_sentence = sent

    if best_sentence:
        return best_sentence.strip()

    # Fallback: first part of the first doc
    first = docs[0].page_content[:400].replace("\n", " ")
    return first + ("..." if len(first) == 400 else "")


# ---------- Pydantic request/response models ----------

class AskRequest(BaseModel):
    question: str
    top_k: int | None = None


class Source(BaseModel):
    source: str
    snippet: str


class AnswerResponse(BaseModel):
    answer: str
    sources: List[Source]


# ---------- FastAPI app ----------

app = FastAPI(
    title="LangChain + FastAPI Q&A",
    description="Semantic search and simple Q&A over local documents using FAISS and HuggingFace embeddings.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load vector store once at startup
_vectordb = load_vectorstore()
logger.info("Vector store ready.")


@app.get("/")
def root():
    return {"status": "ok", "message": "LangChain + FastAPI Q&A is running."}


@app.post("/ask", response_model=AnswerResponse)
def ask(body: AskRequest):
    """
    Main Q&A endpoint:
    - retrieves relevant chunks with FAISS
    - generates a simple local answer
    - returns the answer and source snippets
    """
    start = time.time()

    k = body.top_k or settings.top_k_default
    docs = _vectordb.similarity_search(body.question, k=k)

    answer_text = simple_answer(body.question, docs)

    sources: List[Source] = []
    for d in docs:
        source_name = d.metadata.get("source", "unknown")
        snippet = d.page_content[:300].replace("\n", " ")
        sources.append(Source(source=source_name, snippet=snippet))

    elapsed = time.time() - start
    logger.info("Q: %s | k=%d | time=%.2fs", body.question, k, elapsed)

    return AnswerResponse(answer=answer_text, sources=sources)
