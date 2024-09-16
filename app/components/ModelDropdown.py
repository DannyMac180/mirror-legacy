from fasthtml.common import *

def ModelDropdown():
    models = ["GPT-3.5", "GPT-4", "Custom Model"]
    return Div(
        Label("Select AI Model:", For="model-select"),
        Select(
            *[Option(model, value=model) for model in models],
            id="model-select",
            name="model",
            cls="model-dropdown"
        ),
        hx_change="/change-model",
        cls="model-selector"
    )
