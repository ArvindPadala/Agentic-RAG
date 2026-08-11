import time
from pathlib import Path
import gradio as gr
from gemini_helpers import load_chunks_from_s3, embed_and_index_chunks

def make_upload_fn(s3_client, bucket_name, collection):
    def upload_and_index(file_paths, progress=gr.Progress()):
        if not file_paths:
            yield "⚠️ No files selected."
            return
            
        total_files = len(file_paths)
        status_lines = []
        
        def log(msg):
            status_lines.append(msg)
            return "\n".join(status_lines)
            
        yield log(f"🚀 Starting upload of {total_files} file(s)...")
        
        for i, fpath in enumerate(file_paths):
            p = Path(fpath)
            # Use original filename
            filename = p.name 
            
            s3_key = f"input/documents/{filename}"
            yield log(f"[{i+1}/{total_files}] 📤 Uploading to S3: {filename}...")
            s3_client.upload_file(str(p), bucket_name, s3_key)
            
            filename_without_ext = p.stem
            grounding_key = f"output/documents_grounding/{filename_without_ext}_grounding.json"
            
            yield log(f"[{i+1}/{total_files}] ⏳ Waiting for LandingAI ADE to process (this may take 1-3 minutes)...")
            
            # Polling loop
            max_retries = 60 # 60 * 5s = 5 minutes
            found = False
            for attempt in range(max_retries):
                try:
                    s3_client.head_object(Bucket=bucket_name, Key=grounding_key)
                    found = True
                    break
                except Exception:
                    time.sleep(5)
                    progress((attempt + 1) / max_retries, desc=f"Processing {filename}...")
            
            if not found:
                yield log(f"[{i+1}/{total_files}] ❌ Timeout waiting for ADE processing.")
                continue
                
            yield log(f"[{i+1}/{total_files}] ✅ ADE processing complete! Wait 5 seconds for chunks to flush...")
            time.sleep(5)
            
            yield log(f"[{i+1}/{total_files}] 📥 Downloading parsed chunks...")
            chunks = load_chunks_from_s3(
                s3_client=s3_client,
                bucket=bucket_name,
                chunks_prefix=f"output/documents_chunks/{filename_without_ext}_"
            )
            
            yield log(f"[{i+1}/{total_files}] 🧠 Indexing {len(chunks)} chunks into ChromaDB...")
            if chunks:
                embed_and_index_chunks(
                    chunks=chunks,
                    collection=collection,
                    skip_existing=True
                )
            yield log(f"[{i+1}/{total_files}] 🎉 Done indexing {filename}!")
            
        yield log("🏁 All files processed and indexed successfully!")
    return upload_and_index
