from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.memory import ChatMessageHistory
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import phoenix as px
from phoenix.trace.langchain import LangChainInstrumentor
import os
from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain import hub

load_dotenv()

session = px.launch_app()
print(session.url)

LangChainInstrumentor().instrument()

class LangChainProgram:
    def __init__(self, llm_provider):
        self.llm_provider = llm_provider
        self.llm = self.create_llm()
        self.memory = ChatMessageHistory()
        self.retriever = self.load_retriever()
        self.retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
        self.combine_docs_chain = create_stuff_documents_chain(self.llm, self.retrieval_qa_chat_prompt)
        self.retrieval_chain = create_retrieval_chain(self.retriever.as_retriever(search_kwargs={"k": 5}), self.combine_docs_chain)
        
    def load_retriever(self):
        embeddings = OpenAIEmbeddings()
        retriever = Chroma(persist_directory="./chroma_db", embedding_function=embeddings, collection_name="obsidian_docs")
        return retriever
        
    def create_llm(self):
        if self.llm_provider == "lm-studio":
            return ChatOpenAI(base_url="http://localhost:1234/v1", 
                              api_key="lm-studio", 
                              model="bartowski/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-fp16.gguf",
                              streaming=True)
        elif self.llm_provider == "groq":
            return ChatGroq(model_name="llama3-70b-8192",
                            groq_api_key=os.getenv("GROQ_API_KEY"),
                            streaming=True,
                            temperature=0.7)
        elif self.llm_provider == "gpt-4o":
            return ChatOpenAI(model="gpt-4o",
                              api_key=os.getenv("OPENAI_API_KEY"),
                              streaming=True)
        else:
            raise ValueError(f"Invalid LLM provider: {self.llm_provider}")

    def invoke_chat(self, message):
        self.memory.add_user_message(message)
        response = ""
        
        for chunk in self.retrieval_chain.stream({'input': message, 'chat_history': self.memory.messages}):
            # Assuming each chunk is a dictionary and the answer is stored under the 'answer' key
            if 'answer' in chunk:
                answer = chunk['answer']
                response += answer  # Append only the answer to the full response
                yield answer  # Yield only the answer part of the chunk
        # Add the AI response to the chain's memory
        self.memory.add_ai_message(response)