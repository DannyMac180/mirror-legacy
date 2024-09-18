from fasthtml.common import *

def ToolSelector():
    tools = ["Translation", "Summarization", "Sentiment Analysis"]
    return Div(
        Label("Tools:", For="tool-select"),
        Select(
            *[Option(tool, value=tool) for tool in tools],
            id="tool-select",
            name="tool",
            cls="tool-selector-dropdown"
        ),
        hx_change="/select-tool",
        cls="tool-selector"
    )
