"""
Lambda S3 Event Handler for Document Parsing
============================================
This module is designed to run as an AWS Lambda function. It is triggered when
a new PDF is uploaded to the S3 input bucket. It calls the LandingAI ADE API
to parse the document, then stores the extracted text and layout data back to S3.
"""

import os
from config import settings
from utils.logger import get_logger

logger = get_logger('ade_s3_handler')
import json
import boto3
from pathlib import Path
from urllib.parse import unquote_plus
from landingai_ade import LandingAIADE

s3 = boto3.client("s3")

VISION_AGENT_API_KEY = settings.VISION_AGENT_API_KEY
ADE_MODEL = settings.ADE_MODEL
INPUT_FOLDER = settings.INPUT_FOLDER
OUTPUT_FOLDER = settings.OUTPUT_FOLDER
FORCE_REPROCESS = settings.FORCE_REPROCESS

client = LandingAIADE(apikey=VISION_AGENT_API_KEY)

def ensure_s3_folders(bucket: str) -> None:
    for folder in [INPUT_FOLDER, OUTPUT_FOLDER]:
        try:
            s3.put_object(Bucket=bucket, Key=folder)
            logger.info(f"✅ Ensured folder exists: s3://{bucket}/{folder}")
        except Exception as e:
            logger.info(f"⚠️ Could not ensure folder {folder}: {e}")

def ade_handler(event: dict, context) -> dict:
    """
    AWS Lambda handler for automatically parsing documents uploaded to S3/input/
    and saving Markdown results to S3/output/ with preserved folder structure.
    
    File Organization:
    - input/documents/doc.pdf → 
        - output/documents/doc.md (markdown)
        - output/documents_grounding/doc_grounding.json (visual data)
        - output/documents_chunks/doc_*.json (individual chunks)

    The subfolder inside `input/` is preserved. 
    - documents, financial_reports, legal, etc.
    - invoices, invoice_data, etc.
    - Any custom folder structure
    """
    results = []

    for record in event.get("Records", []):
        bucket = record["s3"]["bucket"]["name"]
        key = unquote_plus(record["s3"]["object"]["key"])
        
        # Skip folder creation events
        if key.endswith("/"):
            logger.info(f"⏩ Skipping folder: {key}")
            continue
            
        doc_id = os.path.basename(key)
        
        # Skip if no filename
        if not doc_id:
            logger.info(f"⏩ Skipping empty filename: {key}")
            continue

        logger.info(f"🚀 Lambda triggered for new upload: {doc_id}")
        ensure_s3_folders(bucket)

        if not key.startswith(INPUT_FOLDER):
            logger.info(f"⏩ Skipping non-input file: {key}")
            continue

        # Extract relative path from input folder to preserve folder structure
        relative_path = key[len(INPUT_FOLDER):] if key.startswith(INPUT_FOLDER) else key
        
        # Get the directory structure and filename
        path_parts = Path(relative_path)
        subfolder = str(path_parts.parent) if path_parts.parent != Path('.') else ''
        filename = path_parts.name
        
        # Remove the original extension (e.g., .pdf) and add .md
        # This converts "document.pdf" to "document.md" instead of "document.pdf.md"
        filename_without_ext = Path(filename).stem  # Gets filename without extension
        
        # Build output key preserving folder structure
        if subfolder and subfolder != '.':
            output_key = f"{OUTPUT_FOLDER}{subfolder}/{filename_without_ext}.md"
        else:
            output_key = f"{OUTPUT_FOLDER}{filename_without_ext}.md"

        # Check if output file already exists (unless force reprocess is enabled)
        if not FORCE_REPROCESS:
            try:
                s3.head_object(Bucket=bucket, Key=output_key)
                logger.info(f"⏭️ Skipping {doc_id} - already processed (output exists: {output_key})")
                results.append({
                    "source": f"s3://{bucket}/{key}",
                    "output": f"s3://{bucket}/{output_key}",
                    "status": "skipped",
                    "reason": "already_processed"
                })
                continue
            except s3.exceptions.ClientError:
                # File doesn't exist, proceed with processing
                pass

        try:
            logger.info(f"📥 Fetching s3://{bucket}/{key}")
            obj = s3.get_object(Bucket=bucket, Key=key)
            file_bytes = obj["Body"].read()

            tmp_path = Path("/tmp") / filename
            tmp_path.write_bytes(file_bytes)

            # Start parsing
            logger.info(f"🤖 Starting ADE parsing for {doc_id} (model={ADE_MODEL})")
            response = client.parse(document=tmp_path, model=ADE_MODEL)
            markdown = response.markdown
            logger.info(f"✅ Finished parsing document: {doc_id}")

            logger.info(f"⬆️ Uploading parsed Markdown → s3://{bucket}/{output_key}")
            if subfolder and subfolder != '.':
                logger.info(f"   Preserved folder structure: {subfolder}/")
            s3.put_object(
                Bucket=bucket,
                Key=output_key,
                Body=markdown.encode("utf-8"),
                ContentType="text/markdown"
            )
            
            # Save grounding data (visual references) in separate folder
            # Use path-based approach for consistent folder structure
            path_parts = Path(output_key).parts
            
            if len(path_parts) >= 2:
                # Extract base folder structure (e.g., 'output/documents' or 'output/financial_reports')
                base_folder = str(Path(*path_parts[:2]))  # First two parts: output/foldername
                relative_path = Path(*path_parts[2:]) if len(path_parts) > 2 else Path(path_parts[-1])
                
                # Create parallel folders with consistent naming
                grounding_folder = f"{base_folder}_grounding"
                chunks_folder = f"{base_folder}_chunks/"
                
                # Build the grounding key path
                grounding_filename = str(relative_path).replace('.md', '_grounding.json')
                grounding_key = str(Path(grounding_folder) / grounding_filename)
            else:
                # Fallback for files directly in output/ (shouldn't happen normally)
                grounding_key = output_key.replace('.md', '_grounding.json')
                chunks_folder = 'output/chunks/'
            try:
                # Parse and properly format grounding data
                chunks_data = []
                if hasattr(response, 'chunks'):
                    for chunk in response.chunks:
                        # Parse chunk data - handle both object and dict formats
                        if hasattr(chunk, '__dict__'):
                            chunk_dict = {
                                'id': getattr(chunk, 'id', ''),
                                'type': getattr(chunk, 'type', ''),
                                'markdown': getattr(chunk, 'markdown', ''),
                            }
                            if hasattr(chunk, 'grounding'):
                                grounding = chunk.grounding
                                if hasattr(grounding, 'page') and hasattr(grounding, 'box'):
                                    box = grounding.box
                                    chunk_dict['grounding'] = {
                                        'page': grounding.page,
                                        'box': {
                                            'left': getattr(box, 'left', 0),
                                            'top': getattr(box, 'top', 0),
                                            'right': getattr(box, 'right', 0),
                                            'bottom': getattr(box, 'bottom', 0)
                                        }
                                    }
                        else:
                            chunk_dict = chunk
                        chunks_data.append(chunk_dict)
                
                splits_data = []
                if hasattr(response, 'splits'):
                    for split in response.splits:
                        if hasattr(split, '__dict__'):
                            split_dict = {
                                'chunks': getattr(split, 'chunks', []),
                                'pages': getattr(split, 'pages', []),
                                'markdown': getattr(split, 'markdown', ''),
                                'class_': getattr(split, 'class_', '')
                            }
                        else:
                            split_dict = split
                        splits_data.append(split_dict)
                
                metadata_data = {}
                if hasattr(response, 'metadata'):
                    metadata = response.metadata
                    if hasattr(metadata, '__dict__'):
                        metadata_data = {
                            'filename': getattr(metadata, 'filename', ''),
                            'page_count': getattr(metadata, 'page_count', 0),
                            'version': getattr(metadata, 'version', ''),
                            'job_id': getattr(metadata, 'job_id', ''),
                            'org_id': getattr(metadata, 'org_id', ''),
                            'credit_usage': getattr(metadata, 'credit_usage', 0),
                            'duration_ms': getattr(metadata, 'duration_ms', 0)
                        }
                    else:
                        metadata_data = metadata
                
                grounding_data = {
                    'chunks': chunks_data,
                    'splits': splits_data,
                    'metadata': metadata_data
                }
                
                # Only save if we have actual chunk data
                if grounding_data['chunks']:
                    logger.info(f"📍 Uploading visual grounding data → s3://{bucket}/{grounding_key}")
                    logger.info(f"   Found {len(grounding_data['chunks'])} chunks with grounding info")
                    
                    # Save as clean JSON
                    s3.put_object(
                        Bucket=bucket,
                        Key=grounding_key,
                        Body=json.dumps(grounding_data, indent=2).encode("utf-8"),
                        ContentType="application/json"
                    )
                    logger.info(f"✅ Saved grounding data: {grounding_key}")
                    
                    # Create individual chunk JSON files for Knowledge Base
                    logger.info(f"📦 Creating individual chunk files for Knowledge Base...")
                    chunk_count = 0
                    for chunk in chunks_data:
                        chunk_id = chunk.get('id', '')
                        if not chunk_id:
                            continue
                        
                        # Extract bbox from grounding
                        grounding = chunk.get('grounding', {})
                        box = grounding.get('box', {})
                        bbox = [
                            box.get('left', 0),
                            box.get('top', 0),
                            box.get('right', 1),
                            box.get('bottom', 1)
                        ]
                        
                        # Create chunk JSON for Knowledge Base
                        chunk_json = {
                            "chunk_id": chunk_id,
                            "chunk_type": chunk.get('type', 'text'),
                            "text": chunk.get('markdown', ''),
                            "bbox": bbox,
                            "page": grounding.get('page', 0),
                            "source_document": filename_without_ext
                        }
                        
                        # Save individual chunk JSON
                        chunk_key = f"{chunks_folder}{filename_without_ext}_{chunk_id}.json"
                        s3.put_object(
                            Bucket=bucket,
                            Key=chunk_key,
                            Body=json.dumps(chunk_json, indent=2).encode("utf-8"),
                            ContentType="application/json"
                        )
                        chunk_count += 1
                    
                    logger.info(f"✅ Created {chunk_count} chunk files in {chunks_folder}")
                else:
                    logger.info(f"⚠️ No chunks found in response for grounding data")
                    
            except Exception as e:
                logger.info(f"⚠️ Could not save grounding data: {e}")

            results.append({
                "source": f"s3://{bucket}/{key}",
                "output": f"s3://{bucket}/{output_key}",
                "status": "success"
            })

            logger.info(f"🎉 Completed pipeline for {doc_id} → {output_key} (clean name: {filename_without_ext}.md)")

        except Exception as e:
            logger.info(f"❌ Error processing {doc_id}: {e}")
            results.append({
                "source": f"s3://{bucket}/{key}",
                "error": str(e),
                "status": "failed"
            })

    logger.info("🏁 All records processed.")
    return {"status": "ok", "results": results}