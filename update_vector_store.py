import os
import json
import hashlib
from datetime import datetime
from langchain_community.document_loaders import ObsidianLoader
import weaviate
import weaviate.classes as wvc
from dotenv import load_dotenv
from google.cloud import logging_v2
from google.auth import default

load_dotenv()

# Path to your Obsidian vault
OBS_VAULT_PATH = os.getenv('OBSIDIAN_PATH')
# Path to the index file
INDEX_FILE_PATH = 'docs_inserted_index.json'
# Weaviate client configuration
WEAVIATE_URL = 'https://mirror-cluster-t3a5zsyf.weaviate.network'

# Your Google Cloud project ID
PROJECT_ID = os.getenv("GCP_PROJECT_ID")

# Set up GCP Cloud Logging
credentials, project = default()
client = logging_v2.Client(project=PROJECT_ID, credentials=credentials)
logger = client.logger('vector-store-update-logs')

def load_index():
    if os.path.exists(INDEX_FILE_PATH):
        try:
            with open(INDEX_FILE_PATH, 'r') as f:
                content = f.read()
                if content.strip():  # Check if file is not empty
                    return json.loads(content)
                else:
                    logger.log_text("Warning: {} is empty. Returning empty dict.".format(INDEX_FILE_PATH), severity="WARNING")
        except json.JSONDecodeError:
            logger.log_text("Error: {} contains invalid JSON. Returning empty dict.".format(INDEX_FILE_PATH), severity="ERROR")
    else:
        logger.log_text("Info: {} does not exist. Returning empty dict.".format(INDEX_FILE_PATH), severity="INFO")
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
        logger.log_text("Warning: File not found: {}".format(file_path), severity="WARNING")
        return None

def get_new_documents(obsidian_loader, index):
    new_docs = []
    for doc in obsidian_loader.load():
        file_path = doc.metadata.get('path')
        if file_path is None:
            logger.log_text("Document has no source path. Skipping. Content preview: {}...".format(doc.page_content[:100]), severity="WARNING")
            continue
        
        try:
            file_hash = get_file_hash(file_path)
            if file_hash not in index:
                index[file_hash] = datetime.now().isoformat()
                new_docs.append(doc)
                logger.log_text("New document found: {}".format(file_path), severity="INFO")
        except Exception as e:
            logger.log_text("Error processing file {}: {}".format(file_path, str(e)), severity="ERROR")
    
    return new_docs

def insert_documents_to_weaviate(client, documents):
    successful_inserts = 0
    failed_inserts = 0
    
    obsidian_docs = client.collections.get("ObsidianDocs")
    with obsidian_docs.batch.dynamic() as batch:
        for doc in documents:
            try:
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
                successful_inserts += 1
                logger.log_text("Successfully inserted document: {}".format(doc.metadata['path']), severity="INFO")
            except Exception as e:
                failed_inserts += 1
                logger.log_text("Failed to insert document {}: {}".format(doc.metadata['path'], str(e)), severity="ERROR")
    
    return successful_inserts, failed_inserts

def main():
    print(f"Using project: {PROJECT_ID}")
    
    try:
        logger.log_text("Starting vector store update process", severity="INFO")
        start_time = datetime.now()

        index = load_index()
        logger.log_text("Loaded index with {} entries".format(len(index)), severity="INFO")

        obsidian_loader = ObsidianLoader(OBS_VAULT_PATH)
        new_documents = get_new_documents(obsidian_loader, index)

        if not new_documents:
            logger.log_text("No new documents found.", severity="INFO")
            return

        logger.log_text("Found {} new documents".format(len(new_documents)), severity="INFO")

        try:
            client = weaviate.connect_to_wcs(
                cluster_url=os.getenv("WEAVIATE_CLUSTER_URL"),
                auth_credentials=weaviate.auth.AuthApiKey(api_key=os.getenv("WEAVIATE_API_KEY")),
                headers={
                    "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")
                }
            )
            logger.log_text("Successfully connected to Weaviate", severity="INFO")
        except Exception as e:
            logger.log_text("Failed to connect to Weaviate: {}".format(str(e)), severity="ERROR")
            return

        # Check if the collection already exists, if not, create it
        if not client.collections.exists("ObsidianDocs"):
            try:
                create_obsidian_docs_collection(client)
                logger.log_text("Created ObsidianDocs collection in Weaviate", severity="INFO")
            except Exception as e:
                logger.log_text("Failed to create ObsidianDocs collection: {}".format(str(e)), severity="ERROR")
                return
        else:
            logger.log_text("ObsidianDocs collection already exists in Weaviate", severity="INFO")

        successful_inserts, failed_inserts = insert_documents_to_weaviate(client, new_documents)

        save_index(index)
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.log_text("Vector store update process completed in {}".format(duration), severity="INFO")
        logger.log_text("Total documents processed: {}".format(len(new_documents)), severity="INFO")
        logger.log_text("Successful insertions: {}".format(successful_inserts), severity="INFO")
        logger.log_text("Failed insertions: {}".format(failed_inserts), severity="INFO")

        print_summary(len(new_documents), successful_inserts, failed_inserts, duration)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        # You might want to log this error to a local file since cloud logging is failing
        with open('error_log.txt', 'a') as f:
            f.write(f"{datetime.now()}: {str(e)}\n")

def create_obsidian_docs_collection(client):
    obsidian_docs = client.collections.create(
        name="ObsidianDocs",
        vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_openai(),
        generative_config=wvc.config.Configure.Generative.openai()
    )
    
    properties = [
        ("content", wvc.data_type.Text()),
        ("source", wvc.data_type.Text()),
        ("path", wvc.data_type.Text()),
        ("created", wvc.data_type.Date()),
        ("last_modified", wvc.data_type.Date()),
        ("last_accessed", wvc.data_type.Date()),
    ]
    
    for name, data_type in properties:
        obsidian_docs.properties.create(name=name, data_type=data_type)

def print_summary(total_docs, successful_inserts, failed_inserts, duration):
    print("\n--- Vector Store Update Summary ---")
    print("Total documents processed: {}".format(total_docs))
    print("Successful insertions: {}".format(successful_inserts))
    print("Failed insertions: {}".format(failed_inserts))
    print("Duration: {}".format(duration))
    print("Logs available in GCP Cloud Logging")
    print("----------------------------------")

if __name__ == "__main__":
    main()