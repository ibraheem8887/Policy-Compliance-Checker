import json
import pandas as pd
import time
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from google.api_core.exceptions import ResourceExhausted

INDEX_PATH = "../data/faiss_index"
RULES_PATH = "../data/rules.json"
OUTPUT = "compliance_results.csv"

MAX_RETRIES = 3      
RETRY_WAIT = 60      

def load_vectorstore():
    return FAISS.load_local(
        INDEX_PATH,
        embeddings=None,
        allow_dangerous_deserialization=True
    )

def load_rules():
    with open(RULES_PATH) as f:
        return json.load(f)

def create_chain(vectorstore):
    # Limit k to reduce token usage per request
    retriever = vectorstore.as_retriever(k=2)

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-pro")

    template = """
Context:
{context}

Question:
{question}

Provide a compliance check with evidence.
"""
    prompt = PromptTemplate(input_variables=["context", "question"], template=template)
    chain = LLMChain(llm=llm, prompt=prompt)

    return chain, retriever

def run_rule(chain, context, question):
    """Run a single rule with retries if quota exceeded."""
    for attempt in range(MAX_RETRIES):
        try:
            return chain.run(context=context, question=question)
        except ResourceExhausted:
            print(f"[WARN] Quota exceeded. Waiting {RETRY_WAIT} seconds before retry...")
            time.sleep(RETRY_WAIT)
    return "Quota exceeded / could not process rule"

def run_compliance():
    vectorstore = load_vectorstore()
    rules = load_rules()
    chain, retriever = create_chain(vectorstore)

    results = []

    for rule in rules:
        question = rule["rule"]
        docs = retriever.get_relevant_documents(question)
        context = "\n\n".join([d.page_content for d in docs])

        answer = run_rule(chain, context, question)

        results.append({
            "Rule ID": rule["id"],
            "Rule": rule["rule"],
            "Result": answer
        })

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT, index=False)
    print(f"[INFO] Saved report → {OUTPUT}")

if __name__ == "__main__":
    run_compliance()
