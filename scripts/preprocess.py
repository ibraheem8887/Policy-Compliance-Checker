from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def extract_chunks(pdf_path, chunk_size=500, overlap=50):
    reader = PdfReader(pdf_path)

    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    
    chunks = splitter.split_text(full_text)

    # Format the chunks
    result = []
    for i, chunk_text in enumerate(chunks):
        result.append({
            "content": chunk_text,
            "page": i + 1
        })
    
    print(f"Extracted {len(result)} chunks from PDF")
    return result

if __name__ == "__main__":
   
    chunks = extract_chunks("../data/policies.pdf")
   
    for i, chunk in enumerate(chunks[:3]):
        print(f"Chunk {i+1}: {chunk['content'][:100]}...")