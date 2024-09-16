import json
import os
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime
from langchain_community.document_loaders import ObsidianLoader
from langchain.text_splitter import CharacterTextSplitter

# Import necessary functions from update_sid_capsule.py
from utils.update_sid_capsule import (
    load_json_file, save_json_file, log_with_timestamp, retry_operation,
    delete_from_sid, add_to_sid, get_all_sid_items, calculate_file_hash,
    OBSIDIAN_PATH, SID_BASE_URL, SID_API_KEY
)

load_dotenv()

script_dir = Path(__file__).parent.absolute()
sid_cache_path = script_dir / "sid_cache.json"

def get_all_obsidian_files():
    obsidian_loader = ObsidianLoader(OBSIDIAN_PATH)
    all_docs = obsidian_loader.load()
    return {doc.metadata['path']: doc for doc in all_docs}

def split_document(doc):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents([doc])

def reconcile_documents():
    log_with_timestamp("Starting SID capsule reconciliation process")
    start_time = datetime.now()

    # Get all documents from Obsidian
    obsidian_docs = get_all_obsidian_files()
    log_with_timestamp(f"Found {len(obsidian_docs)} documents in Obsidian vault")

    # Get all items from SID
    sid_items = get_all_sid_items()
    sid_uris = {item['uri'] for item in sid_items if 'uri' in item}
    log_with_timestamp(f"Found {len(sid_uris)} documents in SID capsule")

    # Identify documents to add, update, or delete
    to_add_or_update = set(obsidian_docs.keys()) - sid_uris
    to_delete = sid_uris - set(obsidian_docs.keys())

    # Process additions and updates
    successful_updates = 0
    for file_path in to_add_or_update:
        doc = obsidian_docs[file_path]
        split_docs = split_document(doc)
        if process_document(file_path, split_docs):
            successful_updates += 1
        else:
            log_with_timestamp(f"Failed to update document: {file_path}", severity="ERROR")

    # Process deletions
    deleted_count = 0
    for uri in to_delete:
        item_id = next((item['item_id'] for item in sid_items if item['uri'] == uri), None)
        if item_id and delete_from_sid(item_id):
            deleted_count += 1
            log_with_timestamp(f"Deleted document from SID: {uri}")
        else:
            log_with_timestamp(f"Failed to delete document from SID: {uri}", severity="ERROR")

    end_time = datetime.now()
    duration = end_time - start_time

    log_with_timestamp(f"SID capsule reconciliation completed in {duration}")
    log_with_timestamp(f"Total documents processed: {len(to_add_or_update)}")
    log_with_timestamp(f"Successful updates: {successful_updates}")
    log_with_timestamp(f"Failed updates: {len(to_add_or_update) - successful_updates}")
    log_with_timestamp(f"Deleted files: {deleted_count}")

def process_document(source, docs):
    log_with_timestamp(f"Processing document: {source}")
    
    chunks_uploaded = 0
    for i, doc in enumerate(docs, 1):
        if add_to_sid(doc.page_content, doc.metadata):
            chunks_uploaded += 1
        else:
            log_with_timestamp(f"Failed to upload document chunk {i}/{len(docs)} to SID capsule: {source}", severity="ERROR")
            return False
    
    return chunks_uploaded == len(docs)

def main():
    try:
        reconcile_documents()
    except Exception as e:
        log_with_timestamp(f"An error occurred: {str(e)}", severity="ERROR")
        with open('reconciliation_error_log.txt', 'a') as f:
            f.write(f"{datetime.now()}: {str(e)}\n")

if __name__ == "__main__":
    main()