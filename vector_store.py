from langchain_chroma import Chroma
from langchain_community.document_loaders import ObsidianLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain.schema import Document
from langchain_community.docstore import InMemoryDocstore
from langchain.indexes import VectorstoreIndexCreator
from dotenv import load_dotenv
import os

load_dotenv()

def update_vector_store():
    # Load documents from Obsidian
    loader = ObsidianLoader(path=os.getenv("OBSIDIAN_PATH"))
    documents = loader.load()

    # Initialize the record manager
    index_creator = VectorstoreIndexCreator(
        vectorstore_cls=Chroma,
        embedding=OpenAIEmbeddings(),
    )
    record_manager = index_creator.record_manager

    # Compare loaded documents with RecordManager
    new_documents = []
    for doc in documents:
        hash = doc.metadata["source"] + doc.page_content
        if not record_manager.has_doc(hash):
            new_documents.append(doc)

    # Process new documents
    if new_documents:
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(new_documents)
        db = Chroma.from_documents(docs, OpenAIEmbeddings(), persist_directory="./chroma_db")
        record_manager.add_documents(new_documents)

    print(f"Added {len(new_documents)} new documents to the vector store.")
    
# Run the function to update the vector store
update_vector_store()

# Schedule the job to run periodically (e.g., every 24 hours)
# Use a scheduler like `cron` or `apscheduler` to run `update_vector_store()`