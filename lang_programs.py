from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from contextlib import contextmanager
from langchain.memory import ChatMessageHistory
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

class LangChainProgram:
    def __init__(self):
        self.chat = None
        self.memory = ChatMessageHistory()
        self.retriever = self.load_retriever()
        
    def load_retriever(self):
        embeddings = OpenAIEmbeddings()
        retriever = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
        return retriever
        
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
            self.memory.add_user_message(message)
            docs = self.retriever.similarity_search(message, k=5)
            messages = [
                SystemMessage(content="You are a helpful, smart, kind, and efficient AI assistant."),
                *self.memory.messages,
                *[HumanMessage(content=doc.page_content) for doc in docs]
            ]
            response = ""
            for chunk in chat.stream(messages):
                response += chunk.content
                yield response
            self.memory.add_ai_message(response)