from google import genai
import os
from config import settings

client = genai.Client(api_key=settings.GEMINI_API_KEY)
models = client.models.list()
for m in models:
    if "flash" in m.name.lower():
        print(m.name)
