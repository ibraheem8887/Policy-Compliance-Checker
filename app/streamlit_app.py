import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv

# Load up our environment stuff
load_dotenv()
google_api_key = os.environ.get("GOOGLE_API_KEY")

# Where we stored our search index
INDEX_PATH = "../data/faiss_index"

@st.cache_resource
def setup_embeddings():
    """Get the embedding model ready - this helps find similar text"""
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        api_key=google_api_key
    )

@st.cache_resource
def load_policy_search():
    """Load up our policy search engine"""
    try:
        embeddings = setup_embeddings()
        search_engine = FAISS.load_local(
            INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        st.success("✓ Policy database loaded and ready!")
        return search_engine
    except Exception as e:
        st.error(f"Oops! Had trouble loading the policy database: {e}")
        return None

@st.cache_resource
def setup_ai_helper():
    """Get our AI assistant ready to answer questions"""
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",  # Using the faster model
        temperature=0,
        api_key=google_api_key
    )

# Set up the page
st.title("Policy Helper 🤔")
st.markdown("Ask me anything about your company policies and I'll find the answers!")

# Load everything we need
policy_search = load_policy_search()
ai_assistant = setup_ai_helper()

# Get the user's question
user_question = st.text_input(
    "What would you like to know about the policies?",
    placeholder="e.g., What's the vacation policy? How do I report an issue?"
)

# When they click the button
if st.button("Find Answer", type="primary") and user_question.strip():
    
    if policy_search is None:
        st.error("Sorry, the policy database isn't working right now. Please try again later.")
    else:
        try:
            # Show we're working on it
            with st.spinner("Searching through policies..."):
                # Find relevant policy sections
                relevant_docs = policy_search.similarity_search(user_question, k=3)
                
                # Combine what we found
                policy_context = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            # Ask the AI to help explain what we found
            with st.spinner("Putting together an answer..."):
                question_prompt = f"""I found these relevant policy sections. Can you help answer the question based on this information?

Here's what the policies say:
{policy_context}

The question is: {user_question}

Please give a clear, helpful answer based only on the policy information above:"""
                
                answer = ai_assistant.invoke(question_prompt)

            # Show the results
            st.subheader("📋 Answer")
            st.write(answer.content)

            # Show where we found the info
            st.subheader("🔍 Where this info came from")
            for i, doc in enumerate(relevant_docs):
                with st.expander(f"Policy section {i+1} (Page {doc.metadata.get('page', 'N/A')})"):
                    st.write(doc.page_content)
                    
        except Exception as e:
            st.error(f"Sorry, something went wrong while searching: {e}")

# Little help text at the bottom
st.markdown("---")
st.caption("💡 Tip: Ask specific questions like 'What is the dress code?' or 'How many sick days do I get?'")