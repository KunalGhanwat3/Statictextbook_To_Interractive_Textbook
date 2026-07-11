def chunk_text(pages, chunk_size=800, overlap=200):
    """
    Improved chunking with better overlap and size for RAG.
    
    Args:
        pages: List of page dictionaries with 'text' and 'page' keys
        chunk_size: Target size for each chunk (increased from 500 to 800)
        overlap: Number of characters to overlap between chunks (increased from 100 to 200)
    
    Returns:
        List of chunk dictionaries with 'content' and 'page' keys
    """
    chunks = []

    for page in pages:
        text = page["text"].strip()

        if not text:
            continue

        # Try to chunk at sentence boundaries for better context
        sentences = text.split('. ')
        
        current_chunk = ""
        current_size = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Add period back if it was removed
            if not sentence.endswith('.'):
                sentence += '.'
            
            sentence_len = len(sentence) + 1  # +1 for space
            
            # If adding this sentence exceeds chunk_size, save current chunk
            if current_size + sentence_len > chunk_size and current_chunk:
                chunks.append({
                    "content": current_chunk.strip(),
                    "page": page["page"]
                })
                
                # Start new chunk with overlap (last few sentences)
                overlap_text = ' '.join(current_chunk.split()[-overlap//5:])  # Rough word overlap
                current_chunk = overlap_text + ' ' + sentence
                current_size = len(current_chunk)
            else:
                current_chunk += ' ' + sentence if current_chunk else sentence
                current_size += sentence_len
        
        # Add the last chunk if it has content
        if current_chunk.strip():
            chunks.append({
                "content": current_chunk.strip(),
                "page": page["page"]
            })

    return chunks