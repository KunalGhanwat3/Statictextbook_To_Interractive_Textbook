from pypdf import PdfReader

def load_pdf(file_path):
    reader = PdfReader(file_path)
    pages=[]

    for i, page in enumerate(reader.pages):
        text = page.extract_text()

        pages.append({
            "text":text,
            "page":i+1
        })
    return pages