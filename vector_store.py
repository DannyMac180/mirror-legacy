import os
import hashlib
import json
import openai
from chromadb.api import API
from chromadb.config import ClientConfig
from langchain.document_loaders import ObsidianLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Initialize Chroma DB
db_dir = './chroma_db'
if not os.path.exists(db_dir):
    os.makedirs(db_dir)

config = ClientConfig(storage_path=db_dir)
client = API(config)

# Create or get the collection named "obsidian_docs"
collection_name = "obsidian_docs"
if collection_name not in client.list_collections():
    client.create_collection(collection_name)
collection = client.get_collection(collection_name)

# Load Obsidian vault documents using LangChain
vault_path = os.getenv('OBSIDIAN_PATH')
loader = ObsidianLoader(vault_path)
documents = loader.load()

# Function to calculate document hash
def calculate_hash(content):
    return hashlib.md5(content.encode()).hexdigest()

# Initialize LangChain text splitter
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

# Initialize LangChain embeddings
openai.api_key = os.getenv('OPENAI_API_KEY')
embeddings = OpenAIEmbeddings()

# Load previously indexed documents
indexed_docs_file = 'indexed_docs.json'
if os.path.exists(indexed_docs_file):
    with open(indexed_docs_file, 'r') as f:
        indexed_docs = json.load(f)
else:
    indexed_docs = {}

# Track current job run documents
current_docs = {}

for doc in documents:
    content = doc.page_content
    file_path = doc.metadata['source']
    
    # Calculate file hash to detect changes
    file_hash = calculate_hash(content)
    if file_path in indexed_docs and indexed_docs[file_path] == file_hash:
        continue
    
    # Split text into chunks
    chunks = text_splitter.split_text(content)
    
    # Embed each chunk
    chunk_embeddings = embeddings.embed_documents(chunks)
    
    for i, embedding in enumerate(chunk_embeddings):
        chunk_id = f"{file_path}:{i}"
        metadata = {'file_path': file_path, 'chunk_index': i}
        collection.upsert(embedding, id=chunk_id, metadata=metadata)
    
    current_docs[file_path] = file_hash

# Save current indexed documents
with open(indexed_docs_file, 'w') as f:
    json.dump(current_docs, f)

# Remove deleted documents from Chroma DB
deleted_docs = set(indexed_docs.keys()) - set(current_docs.keys())
for file_path in deleted_docs:
    chunks = text_splitter.split_text(open(file_path, 'r', encoding='utf-8').read())
    for i in range(len(chunks)):
        chunk_id = f"{file_path}:{i}"
        collection.delete(id=chunk_id)

print("Chroma DB update complete.")
