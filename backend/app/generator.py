import os
from openai import OpenAI

# gpt-4o-mini: cheap, fast, large context, strong reasoning — good default for RAG Q&A.
# Swap to "gpt-4o" if you want max quality and don't mind higher cost.
LLM_MODEL = "gpt-4o-mini"

_client = None


def _get_client():
    """Create the OpenAI client once. Reads OPENAI_API_KEY from the environment."""
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


SYSTEM_PROMPT = (
    "You answer questions using ONLY the provided context from a document. "
    "Be specific and cite concrete names, numbers, and technologies. "
    "If the question asks for a number of items (e.g. 'top 3'), answer as a "
    "numbered list with exactly that many entries. If the context does not "
    "contain the answer, say so plainly."
)


def generate_answer(query, retrieved_docs):
    """Generate an answer with OpenAI, grounded in the retrieved chunks."""
    if not retrieved_docs:
        return {
            "answer": "I could not find relevant information in the document to answer your question.",
            "citations": [],
        }

    context_parts = []
    citations = []
    for idx, doc in enumerate(retrieved_docs):
        page = doc.metadata.get("page", "Unknown")
        citations.append(page)
        context_parts.append(f"[Source {idx+1}, Page {page}]:\n{doc.page_content}\n")
    context = "\n".join(context_parts)

    user_prompt = f"Context:\n{context}\n\nQuestion: {query}"

    try:
        resp = _get_client().chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        answer = resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"LLM generation error: {e}")
        answer = create_fallback_answer(query, retrieved_docs)

    return {
        "answer": answer,
        "citations": sorted(list(set(citations))),
    }


def create_fallback_answer(query, retrieved_docs):
    """Extractive fallback used when the API call fails."""
    import re

    all_text = []
    for doc in retrieved_docs:
        page = doc.metadata.get("page", "Unknown")
        sentences = re.split(r'(?<=[.!?])\s+', doc.page_content)
        for sent in sentences[:5]:
            if len(sent.strip()) > 20:
                all_text.append(f"{sent.strip()} (Page {page})")

    if not all_text:
        return "Based on the retrieved content, I couldn't generate a specific answer. Please try rephrasing your question."

    return " ".join(all_text[:4])