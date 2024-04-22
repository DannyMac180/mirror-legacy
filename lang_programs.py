from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from contextlib import contextmanager
from langchain.memory import ChatMessageHistory

class LangChainProgram:
    def __init__(self):
        self.chat = None
        self.memory = ChatMessageHistory()
        
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
            messages = [
                SystemMessage(content="You are a helpful, smart, kind, and efficient AI assistant."),
                *self.memory.messages
            ]
            response = ""
            for chunk in chat.stream(messages):
                response += chunk.content
                yield response
            self.memory.add_ai_message(response)