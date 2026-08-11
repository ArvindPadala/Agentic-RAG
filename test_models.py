import os
from google import genai

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
for model in client.models.list():
    if "gemini" in model.name:
        print(model.name)
