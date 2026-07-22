from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

def create_vector_store(chunks):
    texts = [chunk["content"] for chunk in chunks]
    metadatas = [{"page": chunk["page"]} for chunk in chunks]

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_texts(texts, embeddings, metadatas=metadatas)

    return vector_store

def retrieve_chunks(query, vector_store, k=5):
    """Return the top-k most similar chunks for the query."""
    return vector_store.similarity_search(query, k=k)