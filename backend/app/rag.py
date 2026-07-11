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
    """
    Retrieve most relevant chunks using semantic similarity.
    
    Args:
        query: User's question
        vector_store: FAISS vector store
        k: Number of chunks to retrieve (increased from 3 to 5 for better context)
    
    Returns:
        List of relevant document chunks
    """
    # Use similarity_search_with_score to get relevance scores
    results_with_scores = vector_store.similarity_search_with_score(query, k=k)
    
    # Filter out chunks with very low similarity (high distance scores)
    # FAISS returns L2 distance, so lower is better
    max_distance = 1.5  # Adjust this threshold based on your needs
    filtered_results = [
        doc for doc, score in results_with_scores 
        if score < max_distance
    ]
    
    # If filtering removed too many, return at least top 3
    if len(filtered_results) < 3 and len(results_with_scores) >= 3:
        filtered_results = [doc for doc, _ in results_with_scores[:3]]
    
    return filtered_results if filtered_results else [doc for doc, _ in results_with_scores]