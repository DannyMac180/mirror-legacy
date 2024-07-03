import os
import json
import hashlib
from datetime import datetime
from langchain.document_loaders import ObsidianLoader
import weaviate
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
        file_path = doc.metadata.get('source')
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

    client = weaviate.Client(
        url=WEAVIATE_URL,
        auth_credentials=weaviate.auth.AuthCredentials(api_key=os.getenv('WEAVIATE_API_KEY')),
        additional_headers={
            "X-OpenAI-Api-Key": os.getenv('OPENAI_API_KEY')  # Make sure this environment variable is set
        }
    )

    insert_documents_to_weaviate(client, new_documents)

    save_index(index)
    print(f"Inserted {len(new_documents)} new documents into Weaviate.")

if __name__ == "__main__":
    main()