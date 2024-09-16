from fastlite import Database
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Message:
    id: int
    sender: str
    content: str
    timestamp: str

# Initialize database
db = Database("data/chat_history.db")
messages = db.create(Message, pk="id")

def add_message(sender: str, content: str):
    message = Message(
        id=messages.next_id(),
        sender=sender,
        content=content,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    messages.insert(message)

def get_messages():
    return messages.all(order_by="id ASC")

def generate_ai_response(message: str) -> str:
    # Placeholder for AI response generation logic
    # Integrate with an AI model based on `current_model`
    response = f"Echo: {message}"  # Replace with actual AI integration
    return response