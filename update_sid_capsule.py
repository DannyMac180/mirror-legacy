from langchain_community.document_loaders import ObsidianLoader
from langchain.text_splitter import CharacterTextSplitter
import json
import os
import requests
import hashlib
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path
from google.cloud import logging_v2
from google.auth import default
from requests_toolbelt.multipart.encoder import MultipartEncoder
import time
from requests.exceptions import Timeout
import urllib.parse

load_dotenv()

# Initialize constants
SID_API_KEY = os.getenv("SID_API_KEY")
CAPSULE_ID = os.getenv("SID_CAPSULE_ID")
SID_BASE_URL = f"https://{CAPSULE_ID}.sid.ai/data"
PROJECT_ID = os.getenv("GCP_PROJECT_ID")
OBSIDIAN_PATH = os.getenv("OBSIDIAN_PATH")

# Set up GCP Cloud Logging
credentials, project = default()
client = logging_v2.Client(project=PROJECT_ID, credentials=credentials)
logger = client.logger('sid-capsule-update-logs')

# Get the directory of the current script
script_dir = Path(__file__).parent.absolute()
hash_db_path = script_dir / "sid_hash_db.json"

def calculate_file_hash(file_path):
    with open(file_path, "rb") as f:
        file_hash = hashlib.md5()
        chunk = f.read(8192)
        while chunk:
            file_hash.update(chunk)
            chunk = f.read(8192)
    return file_hash.hexdigest()

def load_hash_database():
    if hash_db_path.exists():
        with hash_db_path.open("r") as f:
            return json.load(f)
    return {}

def save_hash_database(file_hash_db):
    with hash_db_path.open("w") as f:
        json.dump(file_hash_db, f)

def delete_from_sid(item_id):
    url = f"{SID_BASE_URL}"
    
    headers = {
        "Authorization": f"Bearer {SID_API_KEY}"
    }
    
    params = {
        "item_id": item_id
    }
    
    try:
        response = requests.delete(url, headers=headers, params=params)
        response.raise_for_status()
        log_with_timestamp(f"Successfully deleted item with ID: {item_id}")
        return True
    except requests.exceptions.RequestException as e:
        log_with_timestamp(f"Error deleting item with ID {item_id} from SID: {str(e)}", severity="ERROR")
        return False

def load_documents():
    obsidian_loader = ObsidianLoader(OBSIDIAN_PATH)
    file_hash_db = load_hash_database()
    new_or_modified_docs = []
    current_files = set()
    try:
        print(f"Attempting to load documents from: {OBSIDIAN_PATH}")
        all_docs = obsidian_loader.load()
        print(f"Total documents loaded: {len(all_docs)}")
        for doc in all_docs:
            file_path = doc.metadata.get('path')
            if file_path:
                current_files.add(file_path)
                current_hash = calculate_file_hash(file_path)
                if file_path not in file_hash_db or file_hash_db[file_path] != current_hash:
                    new_or_modified_docs.append(doc)
                    file_hash_db[file_path] = current_hash
                    print(f"New or modified document: {file_path}")
                else:
                    print(f"Unchanged document: {file_path}")
        
        # Check for deleted files only if the hash database file exists
        if hash_db_path.exists():
            deleted_files = set(file_hash_db.keys()) - current_files
            for deleted_file in deleted_files:
                print(f"Deleted document: {deleted_file}")
                if delete_from_sid(deleted_file):
                    del file_hash_db[deleted_file]
                    logger.log_text(f"Deleted document from SID capsule: {deleted_file}", severity="INFO")
                else:
                    logger.log_text(f"Failed to delete document from SID capsule: {deleted_file}", severity="ERROR")
            
            print(f"Deleted documents: {len(deleted_files)}")
            logger.log_text(f"Deleted {len(deleted_files)} documents", severity="INFO")
        else:
            print("No existing hash database found. Skipping deletion step.")
            logger.log_text("No existing hash database found. Skipping deletion step.", severity="INFO")
            deleted_files = set()
        
        save_hash_database(file_hash_db)
        print(f"New or modified documents: {len(new_or_modified_docs)}")
        logger.log_text(f"Loaded {len(new_or_modified_docs)} new or modified documents, out of {len(all_docs)} total", severity="INFO")
        return new_or_modified_docs, len(deleted_files)
    except Exception as e:
        print(f"Error loading documents: {str(e)}")
        logger.log_text(f"Error loading documents from Obsidian vault: {str(e)}", severity="ERROR")
        return [], 0

def split_documents(documents):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(documents)
    logger.log_text(f"Total document chunks after splitting: {len(split_docs)}", severity="INFO")
    if split_docs:
        logger.log_text(f"First chunk content (truncated): {split_docs[0].page_content[:100]}...", severity="INFO")
    else:
        logger.log_text("No document chunks created", severity="WARNING")
    return split_docs

# Add this function to log with timestamps
def log_with_timestamp(message, severity="INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} - {severity}: {message}")
    logger.log_text(message, severity=severity)

# Modify the add_to_sid function
def add_to_sid(content, metadata):
    url = f"{SID_BASE_URL}/file"
    
    # Prepare the multipart form data
    multipart_data = MultipartEncoder(
        fields={
            'file': ('file', content, 'text/plain'),
            'metadata': json.dumps(metadata),
            'time_authored': datetime.now().isoformat() + "Z",
            'uri': metadata.get("source", "")
        }
    )
    
    headers = {
        "Authorization": f"Bearer {SID_API_KEY}",
        "Content-Type": multipart_data.content_type
    }
    
    try:
        log_with_timestamp(f"Uploading chunk to SID: {metadata.get('source', '')}")
        response = requests.post(url, headers=headers, data=multipart_data, timeout=30)  # Add a 30-second timeout
        response.raise_for_status()
        log_with_timestamp(f"Successfully uploaded chunk to SID: {metadata.get('source', '')}")
        return True
    except Timeout:
        log_with_timestamp(f"Timeout while uploading to SID: {metadata.get('source', '')}", severity="ERROR")
        return False
    except requests.exceptions.RequestException as e:
        log_with_timestamp(f"Error uploading to SID: {str(e)}", severity="ERROR")
        return False

# Modify the upload_documents function
def upload_documents(split_docs):
    successful_uploads = 0
    file_hash_db = load_hash_database()
    
    # Group split_docs by their source file
    docs_by_source = {}
    for doc in split_docs:
        source = doc.metadata.get('path')
        if source:
            if source not in docs_by_source:
                docs_by_source[source] = []
            docs_by_source[source].append(doc)
    
    for source, docs in docs_by_source.items():
        log_with_timestamp(f"Processing document: {source}")
        
        # Get the item_id for this document
        item_id = get_item_id_for_document(source)
        
        # Delete the old version from SID if it exists
        if item_id:
            if delete_from_sid(item_id):
                log_with_timestamp(f"Deleted old version of document from SID: {source}")
            else:
                log_with_timestamp(f"Failed to delete old version from SID: {source}", severity="WARNING")
        else:
            log_with_timestamp(f"No existing version found in SID for: {source}")
        
        # Upload all chunks of the new version
        chunks_uploaded = 0
        for i, doc in enumerate(docs, 1):
            log_with_timestamp(f"Uploading chunk {i}/{len(docs)} for document: {source}")
            if add_to_sid(doc.page_content, doc.metadata):
                chunks_uploaded += 1
                log_with_timestamp(f"Uploaded document chunk {i}/{len(docs)} to SID capsule: {source}")
            else:
                log_with_timestamp(f"Failed to upload document chunk {i}/{len(docs)} to SID capsule: {source}", severity="ERROR")
            
            # Add a small delay between uploads to avoid overwhelming the API
            time.sleep(1)
        
        # Only update the hash if all chunks were uploaded successfully
        if chunks_uploaded == len(docs):
            file_hash_db[source] = calculate_file_hash(source)
            successful_uploads += 1
        else:
            log_with_timestamp(f"Not all chunks uploaded successfully for: {source}", severity="WARNING")
    
    # Save the updated hash database
    save_hash_database(file_hash_db)
    return successful_uploads

def get_item_id_for_document(source):
    url = f"https://{CAPSULE_ID}.sid.ai/data"
    headers = {"Authorization": f"Bearer {SID_API_KEY}"}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        items = response.json()

        for item in items:
            if item.get('uri') == source:
                return item.get('item_id')

        # If no matching item is found, return None
        return None

    except requests.exceptions.RequestException as e:
        log_with_timestamp(f"Error fetching items from SID: {str(e)}", severity="ERROR")
        return None

def print_summary(total_chunks, successful_uploads, failed_uploads, deleted_files, duration):
    print("\n--- SID Capsule Update Summary ---")
    print(f"Total document chunks processed: {total_chunks}")
    print(f"Successful uploads: {successful_uploads}")
    print(f"Failed uploads: {failed_uploads}")
    print(f"Deleted files: {deleted_files}")
    print(f"Duration: {duration}")
    print("Logs available in GCP Cloud Logging")
    print("----------------------------------")

# Modify the main function
def main():
    print(f"Using project: {PROJECT_ID}")
    
    try:
        log_with_timestamp("Starting SID capsule update process")
        start_time = datetime.now()

        log_with_timestamp("Loading documents")
        new_or_modified_docs, deleted_files_count = load_documents()
        log_with_timestamp(f"Loaded {len(new_or_modified_docs)} new or modified documents")

        log_with_timestamp("Splitting documents")
        split_docs = split_documents(new_or_modified_docs)
        log_with_timestamp(f"Split into {len(split_docs)} chunks")

        log_with_timestamp("Uploading documents")
        successful_uploads = upload_documents(split_docs)
        log_with_timestamp(f"Uploaded {successful_uploads} chunks successfully")

        end_time = datetime.now()
        duration = end_time - start_time
        
        log_with_timestamp(f"SID capsule update process completed in {duration}")
        log_with_timestamp(f"Total document chunks processed: {len(split_docs)}")
        log_with_timestamp(f"Successful uploads: {successful_uploads}")
        log_with_timestamp(f"Failed uploads: {len(split_docs) - successful_uploads}")
        log_with_timestamp(f"Deleted files: {deleted_files_count}")

        print_summary(len(split_docs), successful_uploads, len(split_docs) - successful_uploads, deleted_files_count, duration)

    except Exception as e:
        log_with_timestamp(f"An error occurred: {str(e)}", severity="ERROR")
        with open('error_log.txt', 'a') as f:
            f.write(f"{datetime.now()}: {str(e)}\n")

if __name__ == "__main__":
    main()