import weaviate
import weaviate.classes as wvc
import os
import requests
import json
from dotenv import load_dotenv
from langchain.document_loaders import ObsidianLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from datetime import datetime

load_dotenv()

client = weaviate.connect_to_weaviate_cloud(
    cluster_url=os.getenv("WEAVIATE_URL"),
    auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
    headers={
        "X-OpenAI-Api-Key": os.environ["OPENAI_API_KEY"]  # Replace with your inference API key
    }
)

try:
    # obsidian_docs = client.collections.create(
    #     name="ObsidianDocs",
    #     vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_openai(),  # If set to "none" you must always provide vectors yourself. Could be any other "text2vec-*" also.
    #     generative_config=wvc.config.Configure.Generative.openai()  # Ensure the `generative-openai` module is used for generative queries
    # )
    
    obsidian_docs = client.collections.get("ObsidianDocs")

    # Load documents from Obsidian vault
    obsidian_path = os.getenv("OBSIDIAN_PATH")
    loader = ObsidianLoader(obsidian_path)
    documents = loader.load()
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    # Prepare data for Weaviate
    obsidian_objects = []
    for doc in splits:
        # Extract metadata
        print(f"Processing document: {doc.metadata.get('source')}")
        # Create object
        obj = {
            "content": doc.page_content,
            "title": doc.metadata.get('source'),
            "created": doc.metadata.get('created'),
            "path": doc.metadata.get('path')
        }
        obsidian_objects.append(obj)

        # Insert data into Weaviate
        obsidian_docs.data.insert_many(obsidian_objects)

        print(f"Inserted {len(obsidian_objects)} documents into Weaviate.")

finally:
    client.close()  # Close client gracefully