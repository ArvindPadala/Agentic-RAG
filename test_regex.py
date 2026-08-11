import re
from app import extract_image_urls, _S3_URL_RE
from agent import run_agent_turn, build_search_tool, create_gemini_client, create_s3_client, load_chroma_collection, build_agent_config
from config import settings

gemini = create_gemini_client(settings.GEMINI_API_KEY)
s3 = create_s3_client()
coll = load_chroma_collection("document_chunks", "./chroma_db")
search_fn, search_tool = build_search_tool(coll, gemini, s3, settings.S3_BUCKET_NAME)
config = build_agent_config(search_tool, {})

resp = run_agent_turn("What were the key findings of the Echinacea study regarding common cold prevention?", [], gemini, config, {"search_knowledge_base": search_fn})
print("--- RAW LLM RESPONSE ---")
print(resp)
print("--- EXTRACTED URLS ---")
print(extract_image_urls(resp))
