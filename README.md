# Interactive Textbook using RAG

This project transforms static textbooks (PDFs) into an interactive learning system using Retrieval-Augmented Generation (RAG).

## 🚀 Features (Planned)
- Upload textbook PDFs
- Ask questions in natural language
- Get answers grounded in textbook content
- Show citations (page numbers)

---

## 🧠 System Architecture

1. PDF Loader → Extract text from PDF
2. Chunking → Split text into smaller pieces
3. Embeddings → Convert text into vectors
4. Vector Database → Store embeddings (indexing)
5. Retrieval → Find relevant chunks
6. LLM → Generate grounded answer

---

## 📂 Project Structure

backend/app/

### main.py
Handles API requests and routes.

### pdf_loader.py
Extracts text from PDF and preserves page numbers for citations.

### chunker.py
Splits extracted PDF text into smaller overlapping chunks for retrieval.

### rag.py
Creates embeddings, stores chunk vectors in FAISS, and retrieves relevant chunks for a query.

### main.py
Connects the full backend pipeline: upload PDF, process it, create vector store, and retrieve relevant chunks for a user query.

---

## 🛠️ Tech Stack
- FastAPI (backend)
- Python
- FAISS (vector database)
- OpenAI / LLM
- React (frontend - later)
