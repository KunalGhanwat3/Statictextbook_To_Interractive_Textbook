from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

_embeddings = None
_reranker = None


def _get_embeddings():
    """Load the sentence-transformer once and reuse it for every upload."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return _embeddings


def _get_reranker():
    """
    Load the cross-encoder reranker once and reuse it. Downloads ~80MB on
    first use, then cached locally. Unlike the bi-encoder embeddings above
    (which pre-compute one vector per chunk), a cross-encoder reads the
    (query, chunk) pair *together* and scores true relevance — slower, so we
    only run it on a small candidate pool, not the whole index.
    """
    # ponytail: ms-marco-MiniLM is the fast proven default; swap to
    # "BAAI/bge-reranker-base" if you want higher accuracy at ~4x the cost.
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


def _to_texts_and_metadatas(chunks):
    texts = [chunk["content"] for chunk in chunks]
    metadatas = [
        {"page": chunk["page"], "source": chunk.get("source")}
        for chunk in chunks
    ]
    return texts, metadatas


def create_vector_store(chunks):
    """Build a fresh FAISS index from a list of chunks."""
    texts, metadatas = _to_texts_and_metadatas(chunks)
    return FAISS.from_texts(texts, _get_embeddings(), metadatas=metadatas)


def add_to_vector_store(vector_store, chunks):
    """
    Append new chunks to an existing FAISS index. If there's no index yet,
    create one. This is what lets several PDFs live in one searchable library.
    """
    if not chunks:
        return vector_store
    if vector_store is None:
        return create_vector_store(chunks)

    texts, metadatas = _to_texts_and_metadatas(chunks)
    vector_store.add_texts(texts, metadatas=metadatas)
    return vector_store


def retrieve_chunks(query, vector_store, k=6, fetch_k=20, rerank=True):
    """
    Return the top-k most relevant chunks for the query, across all documents.

    Two-stage retrieval when rerank=True:
      1. Fast bi-encoder vector search grabs a wide candidate pool (fetch_k).
      2. The cross-encoder re-scores each candidate against the query and we
         keep the true top-k. This fixes the ranking flat cosine search gets
         wrong (near-duplicate chunks, exact-term matches ranked too low).

    rerank=False falls back to plain vector search — the baseline the eval
    harness compares against.
    """
    if not rerank:
        return vector_store.similarity_search(query, k=k)

    candidates = vector_store.similarity_search(query, k=fetch_k)
    if not candidates:
        return []

    pairs = [(query, doc.page_content) for doc in candidates]
    scores = _get_reranker().predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
    return [doc for doc, _ in ranked[:k]]
