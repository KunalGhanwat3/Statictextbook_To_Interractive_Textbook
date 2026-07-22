def chunk_text(pages, chunk_size=800, overlap=200):
    """
    Split each page's text into fixed-size overlapping chunks.

    Splits purely by character length, so every chunk is <= chunk_size even
    when the text has no sentence boundaries (e.g. bulleted resumes). The last
    `overlap` characters are repeated at the start of the next chunk so context
    isn't lost across a boundary.
    """
    chunks = []

    for page in pages:
        text = page["text"].strip()
        if not text:
            continue

        start = 0
        step = max(1, chunk_size - overlap)   # how far to slide the window each time

        while start < len(text):
            chunk = text[start:start + chunk_size].strip()
            if chunk:
                chunks.append({
                    "content": chunk,
                    "page": page["page"],
                })
            start += step

    return chunks