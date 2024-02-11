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
from langchain.chains import RetrievalQA
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
        max_new_tokens = 512,
        temperature = 0.5
    )
    return llm

def retrieval_qa_chain(llm,prompt,db):
    """
    Setting up a retrieval-based question-answering chain,
    and returning response
    """
    qa_chain = RetrievalQA.from_chain_type(llm = llm,
                                           chain_type = 'stuff',
                                           retriever = db.as_retriever(search_kwargs = {'k': 2}),
                                           return_source_documents = True,
                                           chain_type_kwargs = {'prompt':prompt}
                                           )
    
    return qa_chain

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

def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query':query})
    return response

def handle_uploads(uploaded_file):
    file_extension = uploaded_file.name.split(".")[-1].lower()
    try:
        if file_extension == "pdf":
            file_bytes = uploaded_file.read()

            # Use BytesIO to simulate a file-like object
            pdf_reader = PyPDF2.PdfReader(BytesIO(file_bytes))

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

def main():
    st.title("QA Chatbot")

    data_store = "data"

    os.makedirs(data_store, exist_ok =True)

    uploaded_files = st.file_uploader("Upload PDF or Document files", accept_multiple_files=True)

    if uploaded_files:
        with st.spinner("Processing files..."):
            for file in uploaded_files:
                allowed_extensions = {"pdf", "docx"}
                file_extension = file.name.split(".")[-1].lower()

                if file_extension not in allowed_extensions:
                    print(f"{file.name}: Unsupported file type. Please upload a PDF or DOCX file.")
                    continue

                new_file = file.name.split('.')[0]+'.txt'
                text = handle_uploads(file)
                if text:
                    with open(os.path.join(data_store, new_file), "w") as f:
                        f.write(text)
                else:
                    print("Cannot process the document.")

            flag = create_vector_db(data_store, vectorstore_path)
            
            if flag:
                st.success("Vectorstore created successfully!")
            else:
                st.error(f"Error creating vectorstore: {flag}")    

        chain = qa_bot()  # Initialize QA chain

        query = st.text_input("Ask a question:", value="")

        key = 0

        while query:
            with st.spinner("Generating answer..."):
                answer = final_result(query)
            st.write(answer["result"])

            query = st.text_input("Ask another question:", value="", key=key)
            key+=1

    shutil.rmtree(data_store)

if __name__ == "__main__":
    main()