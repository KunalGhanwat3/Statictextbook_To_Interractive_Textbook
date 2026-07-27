from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

_embeddings = None


def _get_embeddings():
    """Load the sentence-transformer once and reuse it for every upload."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return _embeddings


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


def retrieve_chunks(query, vector_store, k=5):
    """Return the top-k most similar chunks for the query, across all documents."""
    return vector_store.similarity_search(query, k=k)
