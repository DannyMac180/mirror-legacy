# import
from langchain_chroma import Chroma
from langchain_community.document_loaders import ObsidianLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from dotenv import load_dotenv
import os

load_dotenv()

# load the document and split it into chunks
loader = ObsidianLoader(path=os.getenv("OBSIDIAN_PATH"))
documents = loader.load()

# split it into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# create the open-source embedding function
embedding_function = OpenAIEmbeddings()

# load it into Chroma
db = Chroma.from_documents(docs, embedding_function, persist_directory="./chroma_db")

# query it
query = "What does Andrej Karpathy say about the Transformer architecture?"
docs = db.similarity_search(query)

# print results
print(docs[0].page_content)