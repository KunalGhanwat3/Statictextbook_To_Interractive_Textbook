from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os

# Use Hugging Face's free inference API
# You can use other models like "google/flan-t5-large", "mistralai/Mistral-7B-Instruct-v0.2"
LLM_MODEL = "google/flan-t5-large"

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
    
    # Enhanced prompt template for better answers
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""Based on the following context from a document, provide a clear, accurate, and comprehensive answer to the question. 

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
    )
    
    try:
        # Initialize LLM (using Hugging Face Hub - free tier)
        # Note: For production, set HUGGINGFACEHUB_API_TOKEN environment variable
        llm = HuggingFaceHub(
            repo_id=LLM_MODEL,
            model_kwargs={
                "temperature": 0.3,  # Lower = more focused, higher = more creative
                "max_length": 512,
                "top_p": 0.95
            }
        )
        
        chain = LLMChain(llm=llm, prompt=prompt_template)
        
        # Generate answer
        answer = chain.run(context=context, question=query)
        
        # Clean up the answer
        answer = answer.strip()
        
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