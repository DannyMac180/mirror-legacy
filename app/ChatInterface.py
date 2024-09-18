from fasthtml.common import *

def ChatInterface():
    return Div(
        Div(
            Ul(id="message-list", cls="message-list"),
            cls="messages-container"
        ),
        Form(
            Group(
                Input(name="user_message", placeholder="Type your message...", cls="chat-input"),
                Button("Send", type="submit", cls="send-button")
            ),
            hx_post="/send-message",
            hx_target="#message-list",
            hx_swap="beforeend",
            cls="chat-form"
        ),
        cls="chat-interface"
    )
