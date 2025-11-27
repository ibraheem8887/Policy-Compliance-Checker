import os
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

from preprocess import extract_chunks

load_dotenv()
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

INDEX_PATH = "../data/faiss_index"

def create_vectorstore(pdf_path, index_path=INDEX_PATH):
    chunks = extract_chunks(pdf_path)
    docs = [
        Document(
            page_content=chunk["content"],
            metadata={"page": chunk["page"]}
        ) for chunk in chunks
    ]

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        api_key=GOOGLE_API_KEY
    )

    vectorstore = FAISS.from_documents(docs, embedding=embeddings)

    vectorstore.save_local(index_path)
    print(f"[INFO] FAISS index saved → {index_path}")

if __name__ == "__main__":
    create_vectorstore("../data/policies.pdf")
