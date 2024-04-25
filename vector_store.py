import weaviate
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import ObsidianLoader
from langchain_community.embeddings.openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

loader = ObsidianLoader(path="/Users/danielmcateer/Library/Mobile Documents/iCloud~md~obsidian/Documents/Ideaverse")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

weaviate_client = weaviate.connect_to_wcs(
    cluster_url=os.getenv("WCS_URL"),
    auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WCS_API_KEY"))
)
db = WeaviateVectorStore.from_documents(
    documents=docs,
    embedding=embeddings,
    client=weaviate_client
)

query = "What is the worldview of the writer according to the documents?"
docs = db.similarity_search(query)


# Print the first 100 characters of each result
for i, doc in enumerate(docs):
    print(f"\nDocument {i+1}:")
    print(doc.page_content[:100] + "...")
