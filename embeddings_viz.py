from nomic import atlas
from langchain.vectorstores import Chroma

CHROMA_DATA_PATH = "chroma_db/"
COLLECTION_NAME = "obsidian_docs"

collection = Chroma.get(collection_name=COLLECTION_NAME, persist_directory=CHROMA_DATA_PATH)
print(collection)

# collection_data = collection.get()

# embeddings = collection_data["embeddings"]
# print(embeddings)

# Assuming 'embeddings' is a numpy array of your embeddings
# dataset = atlas.map_data(embeddings=embeddings)
