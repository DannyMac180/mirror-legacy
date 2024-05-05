from nomic import atlas
import chromadb
from chromadb.utils import embedding_functions

CHROMA_DATA_PATH = "chroma_db/"
COLLECTION_NAME = "obsidian_docs"

client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
collection = client.get_collection(COLLECTION_NAME)

embeddings = collection.get_embeddings()

# Assuming 'embeddings' is a numpy array of your embeddings
dataset = atlas.map_data(embeddings=embeddings)
