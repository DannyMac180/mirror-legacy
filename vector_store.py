import os
import hashlib
import json
import openai
from chromadb import Client, Settings
from langchain.document_loaders import ObsidianLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Initialize Chroma DB
db_dir = './chroma_db'
if not os.path.exists(db_dir):
    os.makedirs(db_dir)

settings = Settings(persist_directory=db_dir)
client = Client(settings=settings)

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
    print(f"Processing document: {file_path}")
    
    # Calculate file hash to detect changes
    file_hash = calculate_hash(content)
    if file_path in indexed_docs and indexed_docs[file_path] == file_hash:
        continue
    
    # Split text into chunks
    chunks = text_splitter.split_text(content)
    
    # Embed each chunk
    chunk_embeddings = embeddings.embed_documents(chunks)
    
    # Add embeddings to ChromaDB
    collection_name = 'obisidan_docs'
    collection = client.get_or_create_collection(name=collection_name)
    for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
        collection.add(
            embeddings=[embedding],
            documents=[chunk],
            ids=[f"{file_path}_chunk_{i}"]
        )
    
    current_docs[file_path] = file_hash

# Save current indexed documents
with open(indexed_docs_file, 'w') as f:
    json.dump(current_docs, f)

print("Chroma DB initialization complete.")
