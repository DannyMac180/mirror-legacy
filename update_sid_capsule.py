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
from requests.exceptions import Timeout, RequestException
from langchain_community.document_loaders import ObsidianLoader
from langchain.text_splitter import CharacterTextSplitter

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
sid_cache_path = script_dir / "sid_cache.json"

MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

def calculate_file_hash(file_path):
    with open(file_path, "rb") as f:
        file_hash = hashlib.md5()
        chunk = f.read(8192)
        while chunk:
            file_hash.update(chunk)
            chunk = f.read(8192)
    return file_hash.hexdigest()

def load_json_file(file_path):
    if file_path.exists():
        with file_path.open("r") as f:
            return json.load(f)
    return {}

def save_json_file(data, file_path):
    with file_path.open("w") as f:
        json.dump(data, f)

def log_with_timestamp(message, severity="INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} - {severity}: {message}")
    logger.log_text(message, severity=severity)

def retry_operation(operation, *args, **kwargs):
    for attempt in range(MAX_RETRIES):
        try:
            return operation(*args, **kwargs)
        except (Timeout, RequestException) as e:
            if attempt == MAX_RETRIES - 1:
                raise
            log_with_timestamp(f"Operation failed, retrying in {RETRY_DELAY} seconds... (Attempt {attempt + 1}/{MAX_RETRIES})", severity="WARNING")
            time.sleep(RETRY_DELAY)

def delete_from_sid(item_id):
    url = f"{SID_BASE_URL}"
    headers = {"Authorization": f"Bearer {SID_API_KEY}"}
    params = {"item_id": item_id}
    
    response = retry_operation(requests.delete, url, headers=headers, params=params)
    response.raise_for_status()
    log_with_timestamp(f"Successfully deleted item with ID: {item_id}")
    return True

def add_to_sid(content, metadata):
    url = f"{SID_BASE_URL}/file"
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
    
    response = retry_operation(requests.post, url, headers=headers, data=multipart_data)
    response.raise_for_status()
    log_with_timestamp(f"Successfully uploaded chunk to SID: {metadata.get('source', '')}")
    return True

def get_all_sid_items():
    url = f"{SID_BASE_URL}"
    headers = {"Authorization": f"Bearer {SID_API_KEY}"}
    
    response = retry_operation(requests.get, url, headers=headers)
    response.raise_for_status()
    return response.json()

def update_sid_cache():
    items = get_all_sid_items()
    sid_cache = {item['uri']: item['item_id'] for item in items if 'uri' in item and 'item_id' in item}
    save_json_file(sid_cache, sid_cache_path)
    return sid_cache

def load_documents():
    obsidian_loader = ObsidianLoader(OBSIDIAN_PATH)
    file_hash_db = load_json_file(hash_db_path)
    new_or_modified_docs = []
    current_files = set()
    
    try:
        all_docs = obsidian_loader.load()
        for doc in all_docs:
            file_path = doc.metadata.get('path')
            if file_path:
                current_files.add(file_path)
                current_hash = calculate_file_hash(file_path)
                if file_path not in file_hash_db or file_hash_db[file_path] != current_hash:
                    new_or_modified_docs.append(doc)
                    file_hash_db[file_path] = current_hash
        
        deleted_files = set(file_hash_db.keys()) - current_files
        for deleted_file in deleted_files:
            del file_hash_db[deleted_file]
        
        save_json_file(file_hash_db, hash_db_path)
        return new_or_modified_docs, deleted_files
    except Exception as e:
        log_with_timestamp(f"Error loading documents: {str(e)}", severity="ERROR")
        return [], set()

def split_documents(documents):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

def process_document(source, docs, sid_cache):
    log_with_timestamp(f"Processing document: {source}")
    
    item_id = sid_cache.get(source)
    if item_id:
        if delete_from_sid(item_id):
            log_with_timestamp(f"Deleted old version of document from SID: {source}")
        else:
            log_with_timestamp(f"Failed to delete old version from SID: {source}", severity="ERROR")
            return False
    
    chunks_uploaded = 0
    for i, doc in enumerate(docs, 1):
        if add_to_sid(doc.page_content, doc.metadata):
            chunks_uploaded += 1
        else:
            log_with_timestamp(f"Failed to upload document chunk {i}/{len(docs)} to SID capsule: {source}", severity="ERROR")
            return False
        time.sleep(1)
    
    return chunks_uploaded == len(docs)

def update_documents(split_docs):
    sid_cache = update_sid_cache()
    file_hash_db = load_json_file(hash_db_path)
    successful_updates = 0
    docs_by_source = {}
    
    for doc in split_docs:
        source = doc.metadata.get('path')
        if source:
            if source not in docs_by_source:
                docs_by_source[source] = []
            docs_by_source[source].append(doc)
    
    for source, docs in docs_by_source.items():
        if process_document(source, docs, sid_cache):
            file_hash_db[source] = calculate_file_hash(source)
            successful_updates += 1
        else:
            log_with_timestamp(f"Failed to update document: {source}", severity="ERROR")
    
    save_json_file(file_hash_db, hash_db_path)
    return successful_updates

def delete_removed_documents(deleted_files, sid_cache):
    deleted_count = 0
    for file in deleted_files:
        item_id = sid_cache.get(file)
        if item_id:
            if delete_from_sid(item_id):
                deleted_count += 1
                log_with_timestamp(f"Deleted removed document from SID: {file}")
            else:
                log_with_timestamp(f"Failed to delete removed document from SID: {file}", severity="ERROR")
    return deleted_count

def main():
    try:
        log_with_timestamp("Starting SID capsule update process")
        start_time = datetime.now()

        new_or_modified_docs, deleted_files = load_documents()
        log_with_timestamp(f"Loaded {len(new_or_modified_docs)} new or modified documents")

        split_docs = split_documents(new_or_modified_docs)
        log_with_timestamp(f"Split into {len(split_docs)} chunks")

        successful_updates = update_documents(split_docs)
        log_with_timestamp(f"Updated {successful_updates} documents successfully")

        sid_cache = load_json_file(sid_cache_path)
        deleted_count = delete_removed_documents(deleted_files, sid_cache)
        log_with_timestamp(f"Deleted {deleted_count} removed documents")

        end_time = datetime.now()
        duration = end_time - start_time
        
        log_with_timestamp(f"SID capsule update process completed in {duration}")
        log_with_timestamp(f"Total documents processed: {len(new_or_modified_docs)}")
        log_with_timestamp(f"Successful updates: {successful_updates}")
        log_with_timestamp(f"Failed updates: {len(new_or_modified_docs) - successful_updates}")
        log_with_timestamp(f"Deleted files: {deleted_count}")

    except Exception as e:
        log_with_timestamp(f"An error occurred: {str(e)}", severity="ERROR")
        with open('error_log.txt', 'a') as f:
            f.write(f"{datetime.now()}: {str(e)}\n")

if __name__ == "__main__":
    main()