from transformers import pipeline
import os

# ponytail: local flan-t5 — HF retired hosted api-inference.huggingface.co, so run in-process.
# base for CPU speed; swap to "google/flan-t5-large" if answer quality matters more than latency.
LLM_MODEL = "google/flan-t5-large"

_pipe = None


def _get_pipe():
    """Lazy-load the seq2seq pipeline once (first call downloads the model)."""
    global _pipe
    if _pipe is None:
        _pipe = pipeline("text2text-generation", model=LLM_MODEL)
    return _pipe


PROMPT_TEMPLATE = """Based on the following context from a document, provide a clear, accurate, and comprehensive answer to the question.

Context:
{context}

Question: {question}

Instructions:
- Answer directly and concisely based ONLY on the provided context
- Include specific details, numbers, technologies, or names mentioned in the context
- If the context mentions projects, list them with their key details
- If the context doesn't contain enough information, say so
- Use professional and clear language

Answer:"""

def generate_answer(query, retrieved_docs):
    """
    Generate answer using LLM with proper context from retrieved documents.
    """
    if not retrieved_docs:
        return {
            "answer": "I could not find relevant information in the document to answer your question.",
            "citations": []
        }
    
    # Extract context and citations
    context_parts = []
    citations = []
    
    for idx, doc in enumerate(retrieved_docs):
        page = doc.metadata.get("page", "Unknown")
        citations.append(page)
        context_parts.append(f"[Source {idx+1}, Page {page}]:\n{doc.page_content}\n")
    
    context = "\n".join(context_parts)
    
    try:
        prompt = PROMPT_TEMPLATE.format(context=context, question=query)

        # flan-t5 caps input at 512 tokens; truncation keeps long contexts from erroring.
        result = _get_pipe()(
            prompt,
            max_new_tokens=256,
            do_sample=False,  # greedy: deterministic, focused answers
            truncation=True,
        )
        answer = result[0]["generated_text"].strip()

        # If answer is too short or looks like an error, provide fallback
        if len(answer) < 10 or "I don't know" in answer.lower():
            answer = create_fallback_answer(query, retrieved_docs)

    except Exception as e:
        print(f"LLM generation error: {e}")
        # Fallback to extractive summarization
        answer = create_fallback_answer(query, retrieved_docs)
    
    return {
        "answer": answer,
        "citations": sorted(list(set(citations)))
    }


def create_fallback_answer(query, retrieved_docs):
    """
    Fallback method using extractive summarization when LLM fails.
    Better than the old keyword matching approach.
    """
    import re
    
    # Combine all context
    all_text = []
    for doc in retrieved_docs:
        page = doc.metadata.get("page", "Unknown")
        sentences = re.split(r'(?<=[.!?])\s+', doc.page_content)
        for sent in sentences[:5]:  # Take top sentences from each chunk
            if len(sent.strip()) > 20:  # Filter out very short fragments
                all_text.append(f"{sent.strip()} (Page {page})")
    
    if not all_text:
        return "Based on the retrieved content, I couldn't generate a specific answer. Please try rephrasing your question."
    
    # Return top relevant sentences
    answer = " ".join(all_text[:4])  # Combine top 4 sentences
    
    return answer