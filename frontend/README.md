# 📚 RAG-Powered Interactive Textbook

An end-to-end full-stack application that transforms static PDF documents into an interactive Q&A system using Retrieval-Augmented Generation (RAG).

---

## 🚀 Features

- Upload any PDF (textbooks, notes, resumes)
- Ask questions in natural language
- Retrieve relevant sections using semantic search
- Generate grounded answers with page-level citations
- Clean React-based UI

---

## 🏗️ Architecture

### Ingestion Pipeline
PDF → Text Extraction → Chunking (500 tokens, 50 overlap) → Embeddings (MiniLM) → FAISS Index

### Query Pipeline
User Query → Query Embedding → Similarity Search (Top-K) → LLM → Answer + Citations

---

## 🧠 Tech Stack

Backend:
- FastAPI (Python)
- FAISS (Vector Database)
- Sentence Transformers (all-MiniLM-L6-v2)
- OpenAI API (LLM for answer generation)

Frontend:
- React (Vite)
- Axios
- CSS

---

## ⚙️ How It Works

1. User uploads a PDF
2. Text is extracted and split into overlapping chunks
3. Chunks are converted into embeddings (384-dimensional vectors)
4. Stored in FAISS for fast semantic retrieval
5. User asks a question
6. Relevant chunks are retrieved using similarity search
7. LLM generates a grounded answer using retrieved context

---

## 📂 Project Structure

interactive-textbook-rag/
├── backend/
│   └── app/
│       ├── main.py
│       ├── pdf_loader.py
│       ├── chunker.py
│       ├── rag.py
│       ├── generator.py
├── frontend/
│   ├── src/
│   └── package.json
├── README.md
└── .gitignore

---

## 🛠️ Setup Instructions

1. Clone the repository

git clone https://github.com/sristi1407/StaticTextbook_to_InteractiveTextbook_using_RAG.git  
cd interactive-textbook-rag

2. Backend setup

pip install -r requirements.txt  

Create a .env file inside backend/ and add:

OPENAI_API_KEY=your_api_key_here

Run backend:

python3 -m uvicorn backend.app.main:app --reload

3. Frontend setup

cd frontend  
npm install  
npm run dev  

---

## 🌐 Usage

1. Open http://localhost:5173  
2. Upload a PDF  
3. Ask questions about the document  
4. Get answers with citations  

---

## 📊 Results

- Reduced hallucination using retrieval-based grounding  
- Improved answer accuracy with contextual chunks  
- Efficient semantic search using FAISS  

---

## 💡 Key Concepts

- Retrieval-Augmented Generation (RAG)  
- Semantic Search  
- Embeddings (MiniLM - 384D)  
- Vector Databases (FAISS)  
- Context-grounded LLM responses  

---

## 🚧 Future Improvements

- Chat-style UI (like ChatGPT)  
- Highlight source text in PDF  
- Multi-document support  
- Streaming responses  
- Deployment (AWS / Vercel)  

---

## 👩‍💻 Author

Sristi Prasad  
MS Computer Science  

---

## ⭐ If you like this project

Give it a star on GitHub!