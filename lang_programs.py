from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.memory import ChatMessageHistory
from langchain.vectorstores import Weaviate
from langchain.embeddings import OpenAIEmbeddings
import weaviate
import phoenix as px
from phoenix.trace.langchain import LangChainInstrumentor
import os
from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain import hub
import json

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
        self.retrieval_chain = create_retrieval_chain(self.retriever, self.combine_docs_chain)
        
    def load_retriever(self):
        client = weaviate.Client(
            url=os.getenv("WEAVIATE_CLUSTER_URL"),
            auth_client_secret=weaviate.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
            additional_headers={
                "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")
            }
        )

        vectorstore = Weaviate(
            client=client,
            index_name="ObsidianNotes",
            text_key="content",
            embedding=OpenAIEmbeddings()
        )

        return vectorstore.as_retriever(search_kwargs={"k": 5})
        
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
            if 'answer' in chunk:
                answer = chunk['answer']
                response += answer
                yield answer
        self.memory.add_ai_message(response)

    def inspect_documents(self):
        results = self.retriever.get_relevant_documents("sample query")
        for i, doc in enumerate(results):
            print(f"Document {i+1}:")
            print(f"Content: {doc.page_content[:500]}...")
            print(f"Metadata: {doc.metadata}")
            print("---")

    def print_schema(self):
        schema = self.retriever.client.schema.get("ObsidianNotes")
        print(json.dumps(schema, indent=2))