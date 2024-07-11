import os
import json
import hashlib
from datetime import datetime
from langchain_community.document_loaders import ObsidianLoader
import weaviate
import weaviate.classes as wvc
from dotenv import load_dotenv

load_dotenv()

# Path to your Obsidian vault
OBS_VAULT_PATH = os.getenv('OBSIDIAN_PATH')
# Path to the index file
INDEX_FILE_PATH = 'docs_inserted_index.json'
# Weaviate client configuration
WEAVIATE_URL = 'https://mirror-cluster-t3a5zsyf.weaviate.network'

def load_index():
    if os.path.exists(INDEX_FILE_PATH):
        try:
            with open(INDEX_FILE_PATH, 'r') as f:
                content = f.read()
                if content.strip():  # Check if file is not empty
                    return json.loads(content)
                else:
                    print(f"Warning: {INDEX_FILE_PATH} is empty. Returning empty dict.")
        except json.JSONDecodeError:
            print(f"Error: {INDEX_FILE_PATH} contains invalid JSON. Returning empty dict.")
    else:
        print(f"Info: {INDEX_FILE_PATH} does not exist. Returning empty dict.")
    return {}

def save_index(index):
    with open(INDEX_FILE_PATH, 'w') as f:
        json.dump(index, f, indent=4)

def get_file_hash(file_path):
    hasher = hashlib.md5()
    try:
        with open(file_path, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()
    except FileNotFoundError:
        print(f"Warning: File not found: {file_path}")
        return None

def get_new_documents(obsidian_loader, index):
    new_docs = []
    for doc in obsidian_loader.load():
        file_path = doc.metadata.get('path')
        if file_path is None:
            print(f"Warning: Document has no source path. Skipping. Content preview: {doc.page_content[:100]}...")
            continue
        
        try:
            file_hash = get_file_hash(file_path)
            if file_hash not in index:
                index[file_hash] = datetime.now().isoformat()
                new_docs.append(doc)
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
    
    return new_docs

def insert_documents_to_weaviate(client, documents):
    for doc in documents:
        client.batch.add_data_object(
            {
                "content": doc.page_content,
                "source": doc.metadata["source"],
                "path": doc.metadata["path"],
                "created": doc.metadata["created"],
                "last_modified": doc.metadata["last_modified"],
                "last_accessed": doc.metadata["last_accessed"],
            },
            "ObsidianDocs"
        )
    client.batch.flush()

def main():
    index = load_index()

    obsidian_loader = ObsidianLoader(OBS_VAULT_PATH)
    new_documents = get_new_documents(obsidian_loader, index)

    if not new_documents:
        print("No new documents found.")
        return

    client = weaviate.connect_to_wcs(
        cluster_url=os.getenv("WEAVIATE_CLUSTER_URL"),  # Replace with your actual WCS cluster URL
        auth_credentials=weaviate.auth.AuthApiKey(api_key=os.getenv("WEAVIATE_API_KEY")),  # Replace with your Weaviate API key
        headers={
            "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")  # Replace with your OpenAI API key
        }
    )
    
    # Check if the collection already exists, if not, create it
    if not client.collections.exists("ObsidianDocs"):
        obsidian_docs = client.collections.create(
            name="ObsidianDocs",
            vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_openai(),
            generative_config=wvc.config.Configure.Generative.openai()
        )
        
        # Add properties to the collection
        obsidian_docs.properties.create(
            name="content",
            data_type=wvc.data_type.Text()
        )
        obsidian_docs.properties.create(
            name="source",
            data_type=wvc.data_type.Text()
        )
        obsidian_docs.properties.create(
            name="path",
            data_type=wvc.data_type.Text()
        )
        obsidian_docs.properties.create(
            name="created",
            data_type=wvc.data_type.Date()
        )
        obsidian_docs.properties.create(
            name="last_modified",
            data_type=wvc.data_type.Date()
        )
        obsidian_docs.properties.create(
            name="last_accessed",
            data_type=wvc.data_type.Date()
        )
        
        print("Created ObsidianDocs collection in Weaviate.")
    else:
        print("ObsidianDocs collection already exists in Weaviate.")
    
    print("Inserting new documents:")
    for i, doc in enumerate(new_documents, 1):
        print(f"{i}. {doc.metadata.get('path', 'Unknown path')}")
    
    insert_documents_to_weaviate(client, new_documents)

    save_index(index)
    print(f"Inserted {len(new_documents)} new documents into Weaviate.")

def insert_documents_to_weaviate(client, documents):
    obsidian_docs = client.collections.get("ObsidianDocs")
    with obsidian_docs.batch.dynamic() as batch:
        for doc in documents:
            batch.add_object(
                properties={
                    "content": doc.page_content,
                    "source": doc.metadata["source"],
                    "path": doc.metadata["path"],
                    "created": doc.metadata["created"],
                    "last_modified": doc.metadata["last_modified"],
                    "last_accessed": doc.metadata["last_accessed"],
                }
            )

if __name__ == "__main__":
    main()