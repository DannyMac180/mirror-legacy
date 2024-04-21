import bs4
from langchain import hub
from langchain_community.document_loaders import ObsidianLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from contextlib import contextmanager

class LangChainProgram:
    def __init__(self):
        self.chat = None
        self.chain = None

    @contextmanager
    def create_chat(self):
        self.chat = ChatOpenAI(base_url="http://localhost:1234/v1", 
                               api_key="lm-studio", 
                               model="bartowski/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-fp16.gguf",
                               streaming=True)
        try:
            yield self.chat
        finally:
            self.chat = None

    def invoke_chat(self, message):
        with self.create_chat() as chat:
            response = chat.invoke([HumanMessage(content=message)])
        return response

    @contextmanager
    def create_chain(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful, smart, kind, and efficient AI assistant. You always fulfill the user's requests to the best of your ability.",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        with self.create_chat() as chat:
            self.chain = prompt | chat
            yield self.chain
        self.chain = None

    def invoke_chain(self, message):
        with self.create_chain() as chain:
            response = chain.invoke([HumanMessage(content=message)])
            for chunk in response:
                yield chunk
# loader = ObsidianLoader(
#     path="/Users/danielmcateer/Library/Mobile Documents/iCloud~md~obsidian/Documents/Ideaverse"
# )

# obsidian_docs = loader.load()

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# splits = text_splitter.split_documents(obsidian_docs)
# vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# # Retrieve and generate using the relevant snippets of the blog.
# retriever = vectorstore.as_retriever()
# prompt = hub.pull("rlm/rag-prompt")

# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)


# rag_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
# )

# print(rag_chain.invoke("What is my wife's name based on my notes?"))

