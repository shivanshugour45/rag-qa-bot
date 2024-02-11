import shutil
import os
import lancedb
import pyarrow as pa
# from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from langchain.document_loaders.text import TextLoader
from langchain.document_loaders.directory import DirectoryLoader
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain.vectorstores.lancedb import LanceDB

# Create a vector database
def create_vector_db(data_path, vectorstore_path):
    try:
        # Check if user wants to override vectorstores with new data or not
        if os.path.exists(vectorstore_path):
            shutil.rmtree(vectorstore_path)
        
        # Load PDF files as chunks of text using PyPDFLoader
        loader = DirectoryLoader(data_path, glob="*.txt", loader_cls=TextLoader, silent_errors=True, loader_kwargs={'encoding': 'utf-8-sig'}, use_multithreading=True)
        documents = loader.load()  # Load documents as text chunks

        # Create text embeddings; numerical vectors that represent the semantics of the text
        embedder = GoogleGenerativeAIEmbeddings(model = 'models/embedding-001')

        # Split text chunks into smaller segments
        # text_splitter = SemanticChunker(embeddings=embedder)
        text_splitter = SentenceTransformersTokenTextSplitter()
        docs = text_splitter.split_documents(documents=documents)

        db = lancedb.connect(vectorstore_path)

        # Define schema (adjust vector size if needed)
        schema = pa.schema([
            pa.field("vector", pa.list_(pa.float32(), 768)),
            pa.field("text", pa.utf8()),
            pa.field("id", pa.utf8()),
        ])

        # Create table
        table = db.create_table("vector_table", schema=schema, mode="overwrite")

        vectorstore = LanceDB(table, embedder)
        
        vectorstore.add_documents(docs)

        return True
    except Exception as e:
        return e

if __name__ == "__main__":
    create_vector_db()