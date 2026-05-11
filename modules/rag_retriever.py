"""RAG retrieval helpers for VSRNet project documents."""

import os

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"

from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_EMBEDDING_MODEL_PATH = Path(
    r"G:\4周学习计划\第三周\VSRNet_RAG_Demo\models\all-MiniLM-L6-v2"
)


def load_documents(doc_folder):
    """Load all UTF-8 txt documents from a folder."""
    doc_folder = Path(doc_folder)
    documents = []

    for file_path in sorted(doc_folder.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8")
        documents.append({"source": file_path.name, "text": text})

    return documents


def split_text(text, chunk_size=500, overlap=80):
    """Split text into overlapping chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size.")

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def build_chunks(documents):
    """Build chunk dictionaries while preserving source metadata."""
    all_chunks = []

    for document in documents:
        chunks = split_text(document["text"])
        for chunk_id, chunk in enumerate(chunks):
            all_chunks.append(
                {
                    "source": document["source"],
                    "chunk_id": chunk_id,
                    "text": chunk,
                }
            )

    return all_chunks


def normalize_embeddings(embeddings):
    """Normalize embeddings for cosine-like inner product search."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.clip(norms, a_min=1e-12, a_max=None)


def build_faiss_index(embeddings):
    """Build a FAISS inner-product index from normalized embeddings."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index


def build_rag_index(doc_folder, model_path=DEFAULT_EMBEDDING_MODEL_PATH):
    """Load documents, create chunks, embed them, and return a RAG index."""
    doc_folder = Path(doc_folder)
    model_path = Path(model_path)

    if not doc_folder.exists():
        raise FileNotFoundError(f"Document folder not found: {doc_folder}")
    if not model_path.exists():
        raise FileNotFoundError(
            "Embedding model path not found. Stage 1 expects a local model at: "
            f"{model_path}"
        )

    print(f"Loading documents from {doc_folder}...")
    documents = load_documents(doc_folder)
    if not documents:
        raise ValueError(f"No .txt documents found in {doc_folder}.")

    print("Splitting documents into chunks...")
    chunks = build_chunks(documents)
    if not chunks:
        raise ValueError("Documents were loaded, but no text chunks were created.")

    print(f"Loaded {len(documents)} documents and created {len(chunks)} chunks.")
    print(f"Loading embedding model from {model_path}...")
    model = SentenceTransformer(str(model_path))

    print("Creating document embeddings...")
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True)
    embeddings = normalize_embeddings(embeddings.astype("float32"))

    print("Building FAISS index...")
    index = build_faiss_index(embeddings)

    return model, index, chunks


def retrieve_project_context(question, model, index, chunks, top_k=3):
    """Retrieve the most relevant project context for a question."""
    if not question.strip():
        raise ValueError("question must not be empty.")
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")
    if not chunks:
        return []

    safe_top_k = min(top_k, len(chunks))
    query_embedding = model.encode([question], convert_to_numpy=True)
    query_embedding = normalize_embeddings(query_embedding.astype("float32"))

    scores, indices = index.search(query_embedding, safe_top_k)

    results = []
    for score, chunk_index in zip(scores[0], indices[0]):
        if chunk_index < 0:
            continue
        chunk = chunks[chunk_index]
        results.append(
            {
                "score": float(score),
                "source": chunk["source"],
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
            }
        )

    return results
