import weaviate
import weaviate.classes as wvc
import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()
print(os.getenv("WCS_URL"))

client = weaviate.connect_to_wcs(
    cluster_url=os.getenv("WCS_URL"),
    auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WCS_API_KEY")),
    headers={
        "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")
    }
)

try:
    obsidian_docs = client.collections.create(
        name="Obsidian_Docs",
        vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_openai(),
        generative_config=wvc.config.Configure.Generative.openai()
    )
finally:
    client.close()

