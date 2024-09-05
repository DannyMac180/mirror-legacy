from langchain.document_loaders import ObsidianLoader
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get the Obsidian vault path from environment variables
obsidian_path = os.getenv("OBSIDIAN_PATH")

# Initialize the ObsidianLoader
loader = ObsidianLoader(obsidian_path)

# Load the documents
documents = loader.load()
print(len(documents))

# Print the first document object for inspection
if documents:
    print("First document object:")
    print(f"Content: {documents[0].page_content[:200]}...")  # Print first 200 characters of content
    print("Metadata:")
    for key, value in documents[0].metadata.items():
        print(f"  {key}: {value}")
else:
    print("No documents were loaded.")
