from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from typing import List
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# Load backend/.env regardless of cwd (uvicorn is launched from repo root)
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from backend.app.pdf_loader import load_pdf
from backend.app.chunker import chunk_text
from backend.app.rag import create_vector_store, add_to_vector_store, retrieve_chunks
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

UPLOAD_FOLDER = "backend/uploads"

# In-memory library shared across requests.
#   vector_store : the combined FAISS index over every uploaded PDF
#   documents    : file name -> {"pages": int, "chunks": [...]}
# The chunks are kept per document so the index can be rebuilt when a
# document is removed or replaced. (This lives in memory, so it resets when
# the server restarts — persisting it to disk is a good future enhancement.)
vector_store = None
documents = {}


def _rebuild_store():
    """Rebuild the combined FAISS index from every document in the library."""
    global vector_store
    all_chunks = []
    for doc in documents.values():
        all_chunks.extend(doc["chunks"])
    vector_store = create_vector_store(all_chunks) if all_chunks else None


def _document_summary():
    """A lightweight view of the library for the frontend."""
    return [
        {"name": name, "pages": doc["pages"], "chunks": len(doc["chunks"])}
        for name, doc in documents.items()
    ]


@app.post("/upload-pdfs")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    global vector_store

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    processed = []
    new_chunks = []
    any_replacement = False

    for file in files:
        name = os.path.basename(file.filename or "")
        if not name.lower().endswith(".pdf"):
            processed.append(
                {"name": file.filename, "pages": 0, "chunks": 0, "skipped": "not a PDF"}
            )
            continue

        file_path = os.path.join(UPLOAD_FOLDER, name)
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        pages = load_pdf(file_path)
        chunks = chunk_text(pages, source=name)

        if name in documents:
            any_replacement = True   # replacing an existing file -> rebuild to avoid dupes
        else:
            new_chunks.extend(chunks)

        documents[name] = {"pages": len(pages), "chunks": chunks}
        processed.append({"name": name, "pages": len(pages), "chunks": len(chunks)})

    if any_replacement:
        _rebuild_store()
    elif new_chunks:
        vector_store = add_to_vector_store(vector_store, new_chunks)

    return {
        "message": "PDF(s) uploaded and processed successfully",
        "processed": processed,
        "documents": _document_summary(),
    }


@app.get("/documents")
def list_documents():
    return {"documents": _document_summary()}


@app.delete("/documents/{name}")
def delete_document(name: str):
    key = os.path.basename(name)
    if key not in documents:
        raise HTTPException(status_code=404, detail="Document not found")
    del documents[key]
    _rebuild_store()
    return {"message": f"Removed {key}", "documents": _document_summary()}


@app.get("/ask")
def ask_question(query: str = Query(...)):
    if vector_store is None:
        return {"error": "Please upload at least one PDF first."}

    retrieved_docs = retrieve_chunks(query, vector_store, k=6)
    result = generate_answer(query, retrieved_docs)

    return {
        "question": query,
        "answer": result["answer"],
        "citations": result["citations"],
    }
