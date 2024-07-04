import weaviate
import weaviate.classes as wvc
from langchain.document_loaders import ObsidianLoader
import os
from dotenv import load_dotenv

load_dotenv()

# Connect to Weaviate cloud
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=os.getenv("WEAVIATE_CLUSTER_URL"),  # Replace with your Weaviate cluster URL
    auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),  # Replace with your Weaviate API key
    headers={
        "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")  # Make sure to set your OpenAI API key as an environment variable
    }
)

# Create a collection using OpenAI embedding and generative models
client.collections.create(
    name="ObsidianNotes",
    description="Collection for Obsidian notes",
    vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_openai(),  # Use OpenAI for embeddings
    generative_config=wvc.config.Configure.Generative.openai(),  # Use OpenAI for generative tasks
)

# Example of how to load and add a document from ObsidianLoader
loader = ObsidianLoader(os.getenv("OBSIDIAN_PATH"))
docs = loader.load()
print(len(docs))

# Prepare batch objects
batch_size = 20  # Adjust this based on your needs and memory constraints
with client.batch.fixed_size(batch_size=batch_size) as batch:
    for i, doc in enumerate(docs):
        properties = {
            "title": doc.metadata.get("source", ""),
            "content": doc.page_content,
            "metadata": {
                "source": doc.metadata.get("source", ""),
                "file_path": doc.metadata.get("path", ""),
                "created_at": doc.metadata.get("created", ""),
                "last_accessed": doc.metadata.get("last_accessed", ""),
            }
        }
        batch.add_object(
            collection="ObsidianNotes",
            properties=properties
        )
        
        # print the range of docs in the entire docs list that were loaded in this batch
        print(f"Loaded docs {i*batch_size} to {(i+1)*batch_size-1}")

client.close()
print("Collection created and documents added successfully!")
