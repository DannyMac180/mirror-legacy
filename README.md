# mirror

Mirror is a tool for mirroring your own mind back to you using AI. The idea is to capture your thoughts and ideas in Obsidian, and then use AI to answer questions about them in way that is insightful to you.

🛠️ Using the following tools and frameworks:

- [LangChain](https://github.com/langchain-ai/langchain) 
- [LM Studio](https://lmstudio.ai/)
- [Obsidian](https://obsidian.md/)
- [Groq](https://groq.com/)
- [Chroma](https://trychroma.com/)

Available LLMs:

- GPT-4o
- LM Studio hosted Llama 3
- Groq hosted Llama 3

It's a work in progress. Currently, it can index your Obsidian vault and answer questions about it using vector search.

See the [lang_programs.py](lang_programs.py) file for the current capabilities of the tool.

See the [vector_store.py](vector_store.py) file for the current capabilities of the tool.

## Prompt Caching Functionality

Mirror now includes prompt caching functionality using LangChain. This feature is integrated into the `ChatAnthropic` class in `lang_programs.py` and the `LangChainProgram` class. The caching mechanism stores and retrieves prompts, reducing redundant API calls and improving performance.

### Benefits of Prompt Caching

- **Reduced Redundant API Calls**: By caching prompts, Mirror reduces the number of API calls made, saving time and resources.
- **Improved Performance**: Cached prompts are retrieved quickly, enhancing the overall performance of the tool.

### Usage Instructions

To take advantage of the prompt caching functionality, follow these steps:

1. Ensure you have the latest version of `lang_programs.py` which includes the prompt caching implementation.
2. When using the `ChatAnthropic` model, prompts will be automatically cached and retrieved as needed.
3. No additional configuration is required; the caching mechanism is built into the `LangChainProgram` class.

For more details, refer to the updated [lang_programs.py](lang_programs.py) file.
