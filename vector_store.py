from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import ObsidianLoader
from langchain.indexes import SQLRecordManager, index
from dotenv import load_dotenv
import os
import chromadb
from chromadb.config import Settings

load_dotenv()

def update_vector_store():
    # Load documents from Obsidian
    loader = ObsidianLoader(path=os.getenv("OBSIDIAN_PATH"))
    documents = loader.load()
    if not documents:
        print("No documents loaded.")
        return

    # Split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    if not docs:
        print("No documents after splitting.")
        return

    # Create an instance of OpenAIEmbeddings
    embeddings = OpenAIEmbeddings()
    if not embeddings:
        print("Embeddings instance not created.")
        return

    # Initialize the Chroma client and collection
    client = chromadb.Client(Settings(persist_directory="./chroma_db"))
    collection = client.get_or_create_collection(name="obsidian_docs")

    collection.add(documents=docs, embeddings=embeddings, ids=[doc["id"] for doc in docs])

    # Retrieve and print embeddings from the collection
    collection_data = collection.get()
    if 'embeddings' not in collection_data:
        print("Embeddings not found in collection data.")
        return
    print(collection_data['embeddings'])
    
    namespace = "chroma/obsidian_docs"  # Use an appropriate namespace
    record_manager = SQLRecordManager(
        namespace, db_url="sqlite:///record_manager_cache.sql"
    )
    record_manager.create_schema()

    # Index the documents using the `index` method
    indexing_result = index(docs, record_manager, collection, cleanup='full')
    print(indexing_result)
    
# Run the function to update the vector store
update_vector_store()

# Schedule the job to run periodically (e.g., every 24 hours)
# Use a scheduler like `cron` or `apscheduler` to run `update_vector_store()`