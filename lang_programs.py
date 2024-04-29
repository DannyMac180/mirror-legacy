from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from contextlib import contextmanager
from langchain.memory import ChatMessageHistory
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

class LangChainProgram:
    def __init__(self, llm_provider):
        self.llm_provider = llm_provider
        self.chat = None
        self.memory = ChatMessageHistory()
        self.retriever = self.load_retriever()
        
    def load_retriever(self):
        embeddings = OpenAIEmbeddings()
        retriever = Chroma(collection_name="obsidian_docs", persist_directory="./chroma_db", embedding_function=embeddings)
        return retriever
        
    @contextmanager
    def create_chat(self):
        if self.llm_provider == "lm-studio":
            self.chat = ChatOpenAI(base_url="http://localhost:1234/v1", 
                                   api_key="lm-studio", 
                                   model="bartowski/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-fp16.gguf",
                                   streaming=True)
        elif self.llm_provider == "groq":
            self.chat = ChatGroq(model_name="groq/llama-3-70b-instruct",
                                 groq_api_key=os.getenv("GROQ_API_KEY"),
                                 streaming=True,
                                 temperature=0.7)
        else:
            raise ValueError(f"Invalid LLM provider: {self.llm_provider}")
        return self.chat

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
            if self.llm_provider == "lm-studio":
                for chunk in chat.stream(messages):
                    response += chunk.content
                    yield response
            elif self.llm_provider == "groq":
                for chunk in chat.agenerate(messages):
                    response += chunk.content
                    yield response
            self.memory.add_ai_message(response)