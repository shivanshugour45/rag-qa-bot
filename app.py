import os
import shutil
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
from langchain.document_loaders.text import TextLoader
from langchain.document_loaders.directory import DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain.vectorstores.lancedb import LanceDB
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain import hub

from io import BytesIO
import PyPDF2  # For PDF handling
from docx import Document
import streamlit as st
import fitz
import validators
import warnings
from datavectoriser import *
warnings.filterwarnings("ignore")

load_dotenv(find_dotenv('.venv\.env')) # read local .env file

vectorstore_path = "vectorstore/db_lancedb" # Get the embeddings path

custom_prompt_template = """Use the following information to answer the users' questions, if you dont know the answer just say "I don't know the answer". DO NOT make up answers that are not based on facts. Explain with detailed answers that are easy to understand. You are free to draw inferences based on the information provided in the context in order to answer the questions as best as possible.

Context: {context}
Question: {question}

Only return the useful aspects of the answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vector store, we also pass in context and question.
    """
    prompt = PromptTemplate(template= custom_prompt_template, input_variables=['context','question'])
    return prompt

def load_llm():
    """
    Loading the llama2 model we have installed using CTransformers
    """
    llm = ChatGoogleGenerativeAI(
        model= "gemini-pro",
        max_output_tokens = 512,
        temperature = 0.5,
        convert_system_message_to_human=True
    )
    return llm

def retrieval_qa_chain(llm,prompt,db):
    """
    Setting up a retrieval-based question-answering chain,
    and returning response
    """
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    combine_docs_chain = create_stuff_documents_chain(
        llm, retrieval_qa_chat_prompt
    )

    retriever = db.as_retriever(search_kwargs = {'k': 2})

    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

    return retrieval_chain

def qa_bot():
    """
    Loading the db and using it in retrieval_qa_chain
    """
    embeddings = GoogleGenerativeAIEmbeddings(model = 'models/embedding-001')
    
    db = lancedb.connect(vectorstore_path)
    table = db.open_table("vector_table")
    vectorstore_db = LanceDB(table, embedding=embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm,qa_prompt,vectorstore_db)
    return qa

@st.cache_resource(show_spinner=False)
def final_result(_qa_result, query):
    response = _qa_result.invoke({'input':query})
    return response

def handle_uploads(uploaded_file):
    file_extension = uploaded_file.name.split(".")[-1].lower()
    try:
        if file_extension == "pdf":
            file_bytes = uploaded_file.read()

            # Use BytesIO to simulate a file-like object
            pdf_reader = PyPDF2.PdfReader(BytesIO(file_bytes))
            # pdf_reader = fitz

            # Extract text from all pages
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"

            return text
        elif file_extension == "docx":
            file_bytes = uploaded_file.read()

            # Use BytesIO to simulate a file-like object
            document = Document(BytesIO(file_bytes))

            # Extract text as before
            text = ""
            for paragraph in document.paragraphs:
                text += paragraph.text + "\n"

            return text
    except Exception as e:
        return False

@st.cache_data(show_spinner=False)
def process_uploaded_files(uploaded_files, data_store):
    for file in uploaded_files:
        allowed_extensions = {"pdf", "docx"}
        file_extension = file.name.split(".")[-1].lower()

        if file_extension not in allowed_extensions:
            print(f"{file.name}: Unsupported file type. Please upload a PDF or DOCX file.")
            continue

        new_file = file.name.split('.')[0]+'.txt'
        text = handle_uploads(file)
        if text:
            with open(os.path.join(data_store, new_file), "w", encoding='utf-8-sig', errors='ignore') as f:
                f.write(text, )
        else:
            print("Cannot process the document.")

    flag = create_vector_db(data_store, vectorstore_path)
    
    if flag:
        return True
    else:
        st.error(f"Error in vectorising the data: {flag}")
        return False


def main():
    data_store = "data"
    os.makedirs(data_store, exist_ok =True)

    # Function to clear the uploaded file data
    def clear_uploaded_file():
        st.session_state.uploaded_file = None

    st.title("QA Chatbot")

    # Initialize session state
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None

    uploaded_files = st.file_uploader("Upload PDF or Document files", accept_multiple_files=True)

    flag = False
    if uploaded_files:
        with st.spinner("Processing uploaded files..."):
            flag = process_uploaded_files(uploaded_files, data_store)
    
    if flag:
        st.subheader("Chat Session")
        chain = qa_bot()  # Initialize QA chain

        query = st.chat_input("Ask your question...", key="query_chat_input")
        user = st.chat_message("user")
        ai = st.chat_message("ai")

        if query:
            user.write(query)
            answer = final_result(chain, query)
            ai.write(answer["answer"])

        if st.button('Reset'):
            st.session_state.clear()
            shutil.rmtree(data_store)
            shutil.rmtree(vectorstore_path)
            clear_uploaded_file()
            uploaded_files = None
            flag = False
            st.rerun()
        
        if st.button('Stop'):
            st.stop()

if __name__ == "__main__":
    main()