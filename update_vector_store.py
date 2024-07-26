import os
import json
import hashlib
import logging
from datetime import datetime
from langchain_community.document_loaders import ObsidianLoader
import weaviate
import weaviate.classes as wvc
from dotenv import load_dotenv
from elasticsearch import Elasticsearch

load_dotenv()

# Path to your Obsidian vault
OBS_VAULT_PATH = os.getenv('OBSIDIAN_PATH')
# Path to the index file
INDEX_FILE_PATH = 'docs_inserted_index.json'
# Weaviate client configuration
WEAVIATE_URL = 'https://mirror-cluster-t3a5zsyf.weaviate.network'

# Set up logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
log_file = f"{log_dir}/update_vector_store_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(filename=log_file, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure Elasticsearch client
es = Elasticsearch(['http://localhost:9200'])

# Add a custom Elasticsearch handler
class ElasticsearchHandler(logging.Handler):
    def emit(self, record):
        doc = {
            'timestamp': datetime.utcnow(),
            'level': record.levelname,
            'message': self.format(record)
        }
        es.index(index="vector-store-logs", document=doc)

# Add the Elasticsearch handler to the logger
es_handler = ElasticsearchHandler()
logger.addHandler(es_handler)

def load_index():
    if os.path.exists(INDEX_FILE_PATH):
        try:
            with open(INDEX_FILE_PATH, 'r') as f:
                content = f.read()
                if content.strip():  # Check if file is not empty
                    return json.loads(content)
                else:
                    logger.warning(f"Warning: {INDEX_FILE_PATH} is empty. Returning empty dict.")
        except json.JSONDecodeError:
            logger.error(f"Error: {INDEX_FILE_PATH} contains invalid JSON. Returning empty dict.")
    else:
        logger.info(f"Info: {INDEX_FILE_PATH} does not exist. Returning empty dict.")
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
        logger.warning(f"Warning: File not found: {file_path}")
        return None

def get_new_documents(obsidian_loader, index):
    new_docs = []
    for doc in obsidian_loader.load():
        file_path = doc.metadata.get('path')
        if file_path is None:
            logger.warning(f"Document has no source path. Skipping. Content preview: {doc.page_content[:100]}...")
            continue
        
        try:
            file_hash = get_file_hash(file_path)
            if file_hash not in index:
                index[file_hash] = datetime.now().isoformat()
                new_docs.append(doc)
                logger.info(f"New document found: {file_path}")
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
    
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
                logger.info(f"Successfully inserted document: {doc.metadata['path']}")
            except Exception as e:
                failed_inserts += 1
                logger.error(f"Failed to insert document {doc.metadata['path']}: {str(e)}")
    
    return successful_inserts, failed_inserts

def main():
    logger.info("Starting vector store update process")
    start_time = datetime.now()

    index = load_index()
    logger.info(f"Loaded index with {len(index)} entries")

    obsidian_loader = ObsidianLoader(OBS_VAULT_PATH)
    new_documents = get_new_documents(obsidian_loader, index)

    if not new_documents:
        logger.info("No new documents found.")
        return

    logger.info(f"Found {len(new_documents)} new documents")

    try:
        client = weaviate.connect_to_wcs(
            cluster_url=os.getenv("WEAVIATE_CLUSTER_URL"),
            auth_credentials=weaviate.auth.AuthApiKey(api_key=os.getenv("WEAVIATE_API_KEY")),
            headers={
                "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")
            }
        )
        logger.info("Successfully connected to Weaviate")
    except Exception as e:
        logger.error(f"Failed to connect to Weaviate: {str(e)}")
        return

    # Check if the collection already exists, if not, create it
    if not client.collections.exists("ObsidianDocs"):
        try:
            create_obsidian_docs_collection(client)
            logger.info("Created ObsidianDocs collection in Weaviate")
        except Exception as e:
            logger.error(f"Failed to create ObsidianDocs collection: {str(e)}")
            return
    else:
        logger.info("ObsidianDocs collection already exists in Weaviate")

    successful_inserts, failed_inserts = insert_documents_to_weaviate(client, new_documents)

    save_index(index)
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    logger.info(f"Vector store update process completed in {duration}")
    logger.info(f"Total documents processed: {len(new_documents)}")
    logger.info(f"Successful insertions: {successful_inserts}")
    logger.info(f"Failed insertions: {failed_inserts}")

    print_summary(len(new_documents), successful_inserts, failed_inserts, duration)

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
    print(f"Total documents processed: {total_docs}")
    print(f"Successful insertions: {successful_inserts}")
    print(f"Failed insertions: {failed_inserts}")
    print(f"Duration: {duration}")
    print(f"Log file: {log_file}")
    print("----------------------------------")

if __name__ == "__main__":
    main()