from fasthtml.common import *

def SidePanel():
    return Div(
        H2("Chats"),
        Ul(
            Li(A("Chat with AI", href="/chat/ai")),
            Li(A("Project Discussion", href="/chat/project")),
            Li(A("General Chat", href="/chat/general")),
        ),
        cls="side-panel"
    )