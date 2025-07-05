import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables from .env file
from pathlib import Path
dotenv_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=dotenv_path)

# Langchain Dependencies
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# --- Configuration & Setup ---

# --- IMPORTANT: API KEY SETUP ---
# This app now requires ONE API key for the chat model via OpenRouter.
os.environ["OPENROUTER_API_KEY"] = "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"

# --- LLM and Embedding Model Initialization ---

# Initialize LLM through OpenRouter
llm = ChatOpenAI(
    temperature=0,
    model_name="deepseek/deepseek-r1-0528-qwen3-8b:free",
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.environ.get("OPENROUTER_API_KEY"),
    model_kwargs={
        "extra_headers": {
            "HTTP-Referer": "http://localhost:8501", # Replace with your app's URL
            "X-Title": "My RAG Chatbot" # Replace with your app's name
        }
    }
)

# Use the same configuration for the condense_question_llm
condense_question_llm = llm

# Initialize HuggingFace Embeddings
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# Setup the app title
st.title('Ask Your PDF (with OpenRouter)')

# --- Document Loading and Vector Store Creation ---

@st.cache_resource(show_spinner="Processing PDF...")
def load_pdf_and_create_vectorstore(pdf_file_path):
    """
    Loads a PDF, splits it into chunks, creates embeddings, and stores them in a Chroma vector database.
    """
    st.write(f"Loading and processing {os.path.basename(pdf_file_path)}...")

    # Load PDF documents
    loader = PyPDFLoader(pdf_file_path)
    documents = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    # Create and return the vector database (ChromaDB)
    # This will create embeddings for the text chunks using OpenAIEmbeddings
    vectorstore = Chroma.from_documents(texts, embedding=embedding)
    st.success("PDF processed and vector database created!")
    return vectorstore

# --- Chat Interface Setup ---

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain" not in st.session_state:
    st.session_state.chain = None

# Allow user to upload a PDF
uploaded_file = st.file_uploader("Upload a PDF to ask questions about", type="pdf")

if uploaded_file is not None:
    # Use the uploaded file's name and size as a key to reset the chain on new file upload
    file_details = f"{uploaded_file.name}-{uploaded_file.size}"
    if st.session_state.get("processed_file") != file_details:
        with st.spinner("Processing new PDF..."):
            # Save the uploaded file temporarily
            with open("uploaded_doc.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
                

            # Load PDF and create vectorstore (cached)
            vectorstore = load_pdf_and_create_vectorstore("uploaded_doc.pdf")

            # Create the Conversational Retrieval Chain
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')
            st.session_state.chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever(),
                memory=memory,
                condense_question_llm=condense_question_llm,
                return_source_documents=True
            )
            st.session_state.processed_file = file_details
            st.session_state.messages = [] # Clear messages for new doc
        st.success("Ready to chat about your PDF!")

# Display historical messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user prompt
prompt = st.chat_input('Ask a question about your document')

if prompt:
    if st.session_state.chain:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.chain({"question": prompt})
                    response = result["answer"]
                    st.markdown(response)

                    # Optionally display source documents
                    if result.get("source_documents"):
                        with st.expander("View Sources"):
                            for doc in result["source_documents"]:
                                page_num = doc.metadata.get('page', 'N/A')
                                if page_num != 'N/A':
                                    page_num += 1
                                st.write(f"**Page {page_num}:** {doc.page_content[:250]}...")
                
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    response = "Sorry, I couldn't process that. Please try again."

            st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.warning("Please upload a PDF first to start the conversation.")
