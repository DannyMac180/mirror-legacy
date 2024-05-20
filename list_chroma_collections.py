import chromadb
from chromadb.config import Settings

CHROMA_DATA_PATH = "chroma_db/"

# Initialize the Chroma client with the persistent directory
client = chromadb.Client(Settings(persist_directory=CHROMA_DATA_PATH))

# List all collections
collections = client.list_collections()

print(collections)

# Print the names of all collections
for collection in collections:
    print(collection.name)

