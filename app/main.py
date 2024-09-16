from fasthtml.common import *
from app.components.ChatInterface import ChatInterface
from app.components.SidePanel import SidePanel
from app.components.ModelDropdown import ModelDropdown
from app.components.ToolSelector import ToolSelector
from app.styles.styles import styles
from app.utils import add_message, get_messages, generate_ai_response

app, rt = fast_app(hdrs=(Style(styles),))

@rt("/")
def index():
    return Title("Discord-Inspired Chatbot"), Container(
        Grid(
            SidePanel(),
            Div(
                ModelDropdown(),
                ToolSelector(),
                ChatInterface(),
                cls="main-chat"
            ),
            cls="main-grid"
        )
    )

@rt("/change-model")
def change_model(model: str):
    # Logic to switch AI models
    current_model = model  # Store the selected model in session or state
    return NotStr(f"<li>{model} model selected.</li>")

@rt("/select-tool")
def select_tool(tool: str):
    # Logic to activate the selected tool
    selected_tool = tool  # Store the selected tool in session or state
    return NotStr(f"<li>{tool} tool selected.</li>")

@rt("/send-message", methods=["POST"])
def send_message(user_message: str):
    add_message(sender="User", content=user_message)
    # Trigger AI response
    ai_response = generate_ai_response(user_message)
    add_message(sender="AI", content=ai_response)
    # Return the AI message to be appended to the message list
    return NotStr(f"<li><strong>AI:</strong> {ai_response}</li>")

serve()