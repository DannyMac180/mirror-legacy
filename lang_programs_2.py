import os
import json
import weaviate
from dotenv import load_dotenv
import openai
import requests

load_dotenv()

class LangProgram:
    def __init__(self, llm_provider):
        self.llm_provider = llm_provider
        self.memory = []
        self.retriever = self.load_retriever()
        
    def load_retriever(self):
        client = weaviate.Client(
            url=os.getenv("WEAVIATE_CLUSTER_URL"),
            auth_client_secret=weaviate.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
            additional_headers={
                "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")
            }
        )
        
        class WeaviateRetriever:
            def __init__(self, client, class_name):
                self.client = client
                self.class_name = class_name

            def get_relevant_documents(self, query):
                response = (
                    self.client.query
                    .get(self.class_name, ["content", "title"])
                    .with_near_text({"concepts": [query]})
                    .with_limit(5)
                    .do()
                )
                documents = []
                for item in response['data']['Get'][self.class_name]:
                    documents.append({
                        "page_content": item['content'],
                        "metadata": {"title": item['title']}
                    })
                return documents

        return WeaviateRetriever(client, "ObsidianNotes")
        
    def create_llm(self):
        if self.llm_provider == "lm-studio":
            return {
                "base_url": "http://localhost:1234/v1",
                "api_key": "lm-studio",
                "model": "bartowski/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-fp16.gguf"
            }
        elif self.llm_provider == "groq":
            return {
                "base_url": "https://api.groq.com/openai/v1",
                "api_key": os.getenv("GROQ_API_KEY"),
                "model": "llama3-70b-8192"
            }
        elif self.llm_provider == "gpt-4o":
            return {
                "base_url": "https://api.openai.com/v1",
                "api_key": os.getenv("OPENAI_API_KEY"),
                "model": "gpt-4"
            }
        else:
            raise ValueError(f"Invalid LLM provider: {self.llm_provider}")

    def invoke_chat(self, message):
        self.memory.append({"role": "user", "content": message})
        relevant_docs = self.retriever.get_relevant_documents(message)
        context = "\n".join([doc["page_content"] for doc in relevant_docs])
        
        llm_config = self.create_llm()
        prompt = f"Context: {context}\n\nHuman: {message}\n\nAI:"
        
        headers = {
            "Authorization": f"Bearer {llm_config['api_key']}",
            "Content-Type": "application/json"
        }
        data = {
            "model": llm_config["model"],
            "messages": self.memory + [{"role": "system", "content": prompt}],
            "stream": True
        }
        
        response = requests.post(
            f"{llm_config['base_url']}/chat/completions",
            headers=headers,
            json=data,
            stream=True
        )
        
        full_response = ""
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line.decode('utf-8').split('data: ')[1])
                if 'choices' in chunk and len(chunk['choices']) > 0:
                    content = chunk['choices'][0]['delta'].get('content', '')
                    if content:
                        full_response += content
                        yield content
        
        self.memory.append({"role": "assistant", "content": full_response})

    def inspect_documents(self):
        results = self.retriever.get_relevant_documents("sample query")
        for i, doc in enumerate(results):
            print(f"Document {i+1}:")
            print(f"Content: {doc['page_content'][:500]}...")
            print(f"Metadata: {doc['metadata']}")
            print("---")

    def print_schema(self):
        schema = self.retriever.client.schema.get("ObsidianNotes")
        print(json.dumps(schema, indent=2))