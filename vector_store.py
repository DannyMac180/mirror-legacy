from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import ObsidianLoader
from langchain.indexes import SQLRecordManager, index
from dotenv import load_dotenv
import os

load_dotenv()

def update_vector_store():
    # Load documents from Obsidian
    loader = ObsidianLoader(path=os.getenv("OBSIDIAN_PATH"))
    documents = loader.load()

    # Split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    # Create an instance of OpenAIEmbeddings
    embeddings = OpenAIEmbeddings()

    # Initialize the vector store
    db = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db", collection_name="obsidian_docs")

    # Initialize the RecordManager
    namespace = "chroma/obsidian_docs"  # Use an appropriate namespace
    record_manager = SQLRecordManager(
        namespace, db_url="sqlite:///record_manager_cache.sql"
    )
    record_manager.create_schema()

    # Index the documents using the `index` method
    index(docs, record_manager, db, cleanup='full')
    
# Run the function to update the vector store
update_vector_store()

# Schedule the job to run periodically (e.g., every 24 hours)
# Use a scheduler like `cron` or `apscheduler` to run `update_vector_store()`