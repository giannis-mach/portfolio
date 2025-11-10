## The Problem I Chose

Modern information retrieval is shifting from keyword search to semantic understanding — systems that can grasp meaning rather than just match words.
I wanted to build a lightweight yet intelligent backend that could search, retrieve, and answer questions from local documents using embeddings, vector similarity, and natural language reasoning.

## What I Did

- Designed a modular architecture separating the pipeline into:

  - build_index.py — responsible for preprocessing and vector index creation.

  - api.py — a deployable FastAPI service for query handling and Q&A.

- Loaded and chunked local documents (.txt, .docx) using LangChain’s DirectoryLoader and RecursiveCharacterTextSplitter.

- Embedded text chunks using Hugging Face Sentence Transformers (all-MiniLM-L6-v2), mapping each chunk into a dense vector space.

- Indexed vectors using FAISS, enabling efficient cosine similarity search across thousands of chunks.

- Implemented a minimal reasoning layer:

  - Extracts relevant sentences by token overlap.

  - Ranks them and generates the most contextually relevant response.

- Built a RESTful FastAPI service with:

  - POST /ask endpoint to submit questions.

  - Automatic JSON schema validation using Pydantic.

  - CORS middleware for safe front-end integration.

- Added clean configuration and logging, allowing adjustable chunk sizes, overlap, and models via environment variables.

## Technical Highlights

- Embedding Model: Sentence-Transformers all-MiniLM-L6-v2

- Vector Database: FAISS (Facebook AI Similarity Search)

- Frameworks: LangChain, FastAPI, Hugging Face Transformers

- Data Sources: Local .txt and .docx files

- Answer Logic: Token-based semantic overlap with sentence-level scoring

## How It Works

- Document Ingestion:
  - The system recursively loads all text and Word files from the /docs directory.

- Chunking & Embedding:
  - Each document is split into overlapping segments to preserve context.
  - These segments are transformed into vector representations using a pretrained transformer model.

- Indexing with FAISS:
  - All embeddings are stored in a FAISS index, enabling sub-second similarity search.

- Question Handling:
  - When a user sends a question to /ask, the API:
      - Searches for the most semantically similar chunks.
      - Scores and selects the best matching sentences.
      - Returns a concise, contextually relevant answer with source snippets.

## The Outcome

- The result is a fully functional retrieval-augmented generation (RAG) backend — capable of answering natural language questions over arbitrary text collections.
It’s lightweight, transparent, and explainable, making it ideal for:
  - Local knowledge base search
  - Company document Q&A bots
  - Research summarisation prototypes

The architecture is framework-agnostic and expandable — new documents can be indexed by simply re-running build_index.py, and the system can later integrate large-language-model (LLM) generation or re-ranking modules if needed.


🛠 Tech Stack

Python · FastAPI · LangChain · HuggingFace Transformers · FAISS · Sentence-Transformers · Pydantic · Uvicorn
