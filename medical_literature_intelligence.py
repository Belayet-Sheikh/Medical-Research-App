# 1. REQUIRED IMPORTS

import os
import streamlit as st

# LangChain and Google specific imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.summarize import load_summarize_chain


# 2. PAGE CONFIGURATION AND STYLING

# --- Page Configuration ---

st.set_page_config(
    page_title="Medical Literature Analysis",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    /* Main app background */
    .main {
        background-color: #f0f2f6; /* Light grey background */
    }
    /* Sidebar styling */
    .st-emotion-cache-16txtl3 {
        background-color: #ffffff;
    }
    /* Button styling */
    .stButton>button {
        border-radius: 8px;
        border: 1px solid #0068c9;
        background-color: #0068c9;
        color: white;
    }
    .stButton>button:hover {
        background-color: #00509e;
        color: white;
        border: 1px solid #00509e;
    }
    /* Expander styling */
    .st-expander {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)



# 3. INITIALIZING API KEY AND MODEL 


# --- Handling API Key ---
if 'GOOGLE_API_KEY' not in os.environ:
    st.error("GOOGLE_API_KEY environment variable not found. The app cannot start.")
    st.stop()

# --- Initialize Models ---
try:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, convert_system_message_to_human=True, max_output_tokens=2048)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
except Exception as e:
    st.error(f"Error initializing Google AI models: {e}")
    st.stop()


# 4. STREAMLIT UI AND APPLICATION LOGIC


# --- Sidebar ---
with st.sidebar:
    st.title("🩺 Analysis Engine")
    st.markdown("Enter the URLs of the medical articles you wish to analyze below.")
    
    with st.form("input_form"):
        url1 = st.text_input("URL 1", key="url1")
        url2 = st.text_input("URL 2", key="url2")
        url3 = st.text_input("URL 3", key="url3")
        process_button = st.form_submit_button(label="Analyze Articles")

# --- Main Page Title ---
st.title("Medical Literature Intelligence")
st.subheader("Your AI-Powered Research Assistant")
st.divider()

# --- Initializing Session State  ---
if "processed" not in st.session_state:
    st.session_state.processed = False
    st.session_state.vector_store = None
    st.session_state.docs = None

# --- Main Processing Logic ---
if process_button:
    urls = [url for url in [url1, url2, url3] if url.strip()]
    if not urls:
        st.sidebar.warning("Please enter at least one valid URL.")
    else:
        with st.spinner("Analyzing content... This may take a moment."):
            try:
                loader = WebBaseLoader(web_paths=urls)
                docs = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
                texts = text_splitter.split_documents(docs)
                if not texts:
                    st.error("Could not extract any text from the provided URLs. Please check the links.")
                else:
                    vector_store = FAISS.from_documents(texts, embedding=embeddings)
                    st.session_state.vector_store = vector_store
                    st.session_state.docs = docs
                    st.session_state.processed = True
                    st.sidebar.success("Analysis complete!")
            except Exception as e:
                st.sidebar.error(f"An error occurred: {e}")
                st.session_state.processed = False

# --- Displaying Results ---
if st.session_state.processed:
    st.header("Analysis Results", anchor=False)

    # --- Summary Section ---
    with st.expander("**Executive Summary of Articles**", expanded=True):
        with st.spinner("Generating summary..."):
            summarize_chain = load_summarize_chain(llm, chain_type="map_reduce")
            summary = summarize_chain.run(st.session_state.docs)
            st.write(summary)

    # --- Q&A Section ---
    with st.expander("**Question & Answer based on Articles**", expanded=True):
        query = st.text_input("Ask a specific question about the content:", placeholder="e.g., What were the primary endpoints of the study?")
        if query:
            with st.spinner("Searching for the answer..."):
                qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
                    llm=llm, chain_type="stuff", retriever=st.session_state.vector_store.as_retriever()
                )
                result = qa_chain({"question": query}, return_only_outputs=True)
                
                st.subheader("Answer", anchor=False)
                st.success(result['answer'])
                
                st.subheader("Sources", anchor=False)
                st.info(result['sources'])
else:
    st.info("Enter article URLs in the sidebar and click 'Analyze Articles' to begin.")
