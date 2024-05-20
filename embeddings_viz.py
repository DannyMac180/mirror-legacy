import chromadb
from chromadb.config import Settings

CHROMA_DATA_PATH = "./chroma_db"
COLLECTION_NAME = "obsidian_docs"

# Initialize the Chroma client with the persistent directory
client = chromadb.Client(persist_directory=CHROMA_DATA_PATH)

# Access the collection (replace 'your_collection_name' with the actual collection name)
collection = client.get_collection(COLLECTION_NAME)

# Use the collection as needed
print(collection)

# collection_data = collection.get()

# embeddings = collection_data["embeddings"]
# print(embeddings)

# Assuming 'embeddings' is a numpy array of your embeddings
# dataset = atlas.map_data(embeddings=embeddings)
