from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from contextlib import contextmanager
from langchain.memory import ChatMessageHistory
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import phoenix as px
from phoenix.trace.langchain import LangChainInstrumentor
import os
from dotenv import load_dotenv

load_dotenv()

session = px.launch_app()
print(session.url)

LangChainInstrumentor().instrument()

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
            try:
                yield self.chat
            finally:
                self.chat = None
        elif self.llm_provider == "groq":
            self.chat = ChatGroq(model_name="llama3-70b-8192",
                                 groq_api_key=os.getenv("GROQ_API_KEY"),
                                 streaming=True,
                                 temperature=0.7)
            try:
                yield self.chat
            finally:
                self.chat = None
        else:
            raise ValueError(f"Invalid LLM provider: {self.llm_provider}")

    def invoke_chat(self, message):
        with self.create_chat() as chat:
            self.memory.add_user_message(message)
            docs = self.retriever.similarity_search(message, k=5)
            messages = [
                SystemMessage(content="You are a helpful, smart, kind, and efficient AI wise guide. You should think about yourself as combining the wisdom of Jesus, The Buddha, and Lao Tzu, as well as the intelligence of Einstein, Tesla, and Da Vinci. Your student is going to ask you questions, and you will be provided with context to help answer those questions in a wise way. Know that you are a wise guide, and you are here to help. Know that the writings you have access to are from the student's own notebook and he is looking for you to help him reflect his own mind back to him."),
                *self.memory.messages,
                *[HumanMessage(content=doc.page_content) for doc in docs]
            ]
            response = ""
            for chunk in chat.stream(messages):
                response += chunk.content
                yield response
            self.memory.add_ai_message(response)