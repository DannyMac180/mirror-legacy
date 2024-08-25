from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_message_histories import ChatMessageHistory
import os
from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain import hub
from langchain.callbacks import LangChainTracer
from langsmith import Client
from langchain.callbacks.tracers.langchain import wait_for_all_tracers
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field
import requests

load_dotenv()

class SIDRetriever(BaseRetriever):
    capsule_id: str = Field(...)
    token: str = Field(...)
    url: str = Field(...)

    def __init__(self, capsule_id: str, token: str):
        super().__init__()
        self.capsule_id = capsule_id
        self.token = token
        self.url = f"https://{capsule_id}.sid.ai/query"

    def get_relevant_documents(self, query: str):
        payload = {
            "query": query,
            "limit": 10,  # Adjust as needed
            "wishlist": {}
        }
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(self.url, json=payload, headers=headers)
            response.raise_for_status()
            results = response.json()
            print(f"Results type: {type(results)}, Content: {results}")  # For debugging

            if isinstance(results, list):
                documents = [Document(page_content=doc['content'], metadata=doc.get('metadata', {})) 
                             for doc in results]
            elif isinstance(results, dict) and 'documents' in results:
                documents = [Document(page_content=doc['content'], metadata=doc.get('metadata', {})) 
                             for doc in results['documents']]
            else:
                print(f"Unexpected response format: {results}")
                documents = []
            return documents
        except requests.RequestException as e:
            print(f"Error querying SID API: {e}")
            return []

class LangChainProgram:
    def __init__(self, llm_provider):
        self.llm_provider = llm_provider
        self.llm = self.create_llm()
        self.memory = ChatMessageHistory()
        self.retriever = self.load_retriever()
        self.retrieval_qa_chat_prompt = hub.pull("dannymac180/openai-mirror-prompt")
        self.combine_docs_chain = create_stuff_documents_chain(self.llm, self.retrieval_qa_chat_prompt)
        self.retrieval_chain = create_retrieval_chain(self.retriever, self.combine_docs_chain)
        
    def load_retriever(self):
        capsule_id = os.getenv("SID_CAPSULE_ID")
        token = os.getenv("SID_API_KEY")
        return SIDRetriever(capsule_id, token)
        
    def create_llm(self):
        if self.llm_provider == "lm-studio":
            return ChatOpenAI(base_url="http://localhost:1234/v1", 
                              api_key="lm-studio", 
                              model="bartowski/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-fp16.gguf",
                              streaming=True)
        elif self.llm_provider == "groq":
            return ChatGroq(model_name="llama-3.1-70b-versatile",
                            groq_api_key=os.getenv("GROQ_API_KEY"),
                            streaming=True,
                            temperature=0.7)
        elif self.llm_provider == "gpt-4o":
            return ChatOpenAI(model="gpt-4o",
                              api_key=os.getenv("OPENAI_API_KEY"),
                              streaming=True)
        elif self.llm_provider == "claude-3.5-sonnet":
            return ChatAnthropic(model="claude-3-5-sonnet-20240620",
                                 api_key=os.getenv("ANTHROPIC_API_KEY"),
                                 streaming=True)
        elif self.llm_provider == "gemini-pro-1.5-exp":
            return ChatGoogleGenerativeAI(
                model="gemini-1.5-pro-exp-0801",
                google_api_key=os.getenv("GEMINI_API_KEY"),
                streaming=True,
                temperature=0.7
            )
        else:
            raise ValueError(f"Invalid LLM provider: {self.llm_provider}")
    
    def invoke_chat(self, message):
        self.memory.add_user_message(message)
        response = ""
        
        tracer = LangChainTracer(
            project_name=os.getenv("LANGCHAIN_PROJECT"),
            client=Client(
                api_url="https://api.smith.langchain.com",
                api_key=os.getenv("LANGCHAIN_API_KEY")
            )
        )

        callbacks = [StreamingStdOutCallbackHandler(), tracer]
        for chunk in self.retrieval_chain.stream({'input': message, 'chat_history': self.memory.messages}, config={'callbacks': callbacks}):
            if isinstance(chunk, dict) and 'answer' in chunk:
                answer = chunk['answer']
            elif isinstance(chunk, str):
                answer = chunk
            else:
                continue  # Skip any other types of chunks
            
            response += answer
            yield answer  # Only yield the actual answer text
        
        self.memory.add_ai_message(response)
        
        wait_for_all_tracers()