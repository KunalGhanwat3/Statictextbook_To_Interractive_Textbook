from fastapi import FastAPI, UploadFile, File, Query
import os
from fastapi.middleware.cors import CORSMiddleware
from backend.app.pdf_loader import load_pdf
from backend.app.chunker import chunk_text
from backend.app.rag import create_vector_store, retrieve_chunks
from backend.app.generator import generate_answer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
vector_store = None


@app.get("/ask")
def ask_question(query: str = Query(...)):
    global vector_store

    if vector_store is None:
        return {"error": "Please upload a PDF first."}

    retrieved_docs = retrieve_chunks(query, vector_store, k=2)
    result = generate_answer(query, retrieved_docs)

    return {
        "question": query,
        "answer": result["answer"],
        "citations": result["citations"]
    }

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    global vector_store

    upload_folder = "backend/uploads"
    os.makedirs(upload_folder, exist_ok=True)

    file_path = os.path.join(upload_folder, file.filename)

    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    pages = load_pdf(file_path)
    chunks = chunk_text(pages)
    vector_store = create_vector_store(chunks)

    return {
        "message": "PDF uploaded and processed successfully",
        "filename": file.filename,
        "total_pages": len(pages),
        "total_chunks": len(chunks)
    }


@app.get("/ask")
def ask_question(query: str = Query(...)):
    global vector_store

    if vector_store is None:
        return {"error": "Please upload a PDF first."}

    results = retrieve_chunks(query, vector_store, k=3)

    response = []
    for doc in results:
        response.append({
            "content": doc.page_content,
            "page": doc.metadata.get("page")
        })

    return {
        "question": query,
        "retrieved_chunks": response
    }