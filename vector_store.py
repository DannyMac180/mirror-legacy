import weaviate
import weaviate.classes as wvc
import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

client = weaviate.connect_to_wcs(
    cluster_url=os.getenv("WCS_URL"),
    auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WCS_API_KEY")),
    headers={
        "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")
    }
)

try:
    pass
finally:
    client.close()

