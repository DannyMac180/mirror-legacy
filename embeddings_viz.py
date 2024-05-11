from nomic import atlas
import chromadb

CHROMA_DATA_PATH = "chroma_db/"
COLLECTION_NAME = "obsidian_docs"

client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
collection = client.get_collection(COLLECTION_NAME)

collection_data = collection.get()

print(collection_data.keys())

embeddings = collection_data["embeddings"]

print(type(embeddings))
print(embeddings)

# Assuming 'embeddings' is a numpy array of your embeddings
dataset = atlas.map_data(embeddings=embeddings)
