import os
import re
import tempfile
from typing import List, Dict, Any

from google.cloud import storage
from langchain_core.documents import Document
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from markitdown import MarkItDown
from pinecone import Pinecone, ServerlessSpec

from excelProcessor import excel_to_document
from utils.chunking import create_documents, clean_text

from dotenv import load_dotenv

load_dotenv()

PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_DIMENSION = 1024
PINECONE_METRIC = "cosine"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"

try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    if not pc.has_index(PINECONE_INDEX_NAME):
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=PINECONE_DIMENSION,
            metric=PINECONE_METRIC,
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
    pinecone_index = pc.Index(PINECONE_INDEX_NAME)
    embeddings_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
    storage_client = storage.Client()
except KeyError as e:
    raise RuntimeError(f"Environment variable {e} is not set. Please set up your .env file.") from e


async def load_file(file_path: str):
    try:
        md = MarkItDown()
        if file_path.lower().endswith(('.txt', '.md')):
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif file_path.lower().endswith(('.xlsx')):
            return await excel_to_document(file_path)
        else:
            result = md.convert(file_path)
            return result.text_content
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return f"Error loading file: {e}"


async def process_file(bucket_name: str, file_name: str, metadata: Dict[str, Any]):
    print(f"--- Starting processing for {file_name} from bucket {bucket_name} ---")

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_name)[1]) as temp_file:
        local_path = temp_file.name

    try:
        print(f"Downloading gs://{bucket_name}/{file_name} to {local_path}...")
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_name)
        blob.download_to_filename(local_path)
        print("Download complete.")

        print("Loading and cleaning content...")
        if not local_path.lower().endswith('.xlsx'):
            raw_content = await load_file(local_path)
            cleaned_content = clean_text(raw_content)

            if not cleaned_content or "Error loading file" in cleaned_content:
                print(f"Skipping file {file_name} due to loading error or empty content.")
                return

            print("Chunking document with legal-aware splitter...")
            legal_docs = create_documents(cleaned_content, f"gs://{bucket_name}/{file_name}")
            print(f"Created {len(legal_docs)} chunks.")

            for d in legal_docs:
                d.metadata.update({
                    **metadata,
                })

            print(f"Upserting {len(legal_docs)} documents to Pinecone index '{PINECONE_INDEX_NAME}'...")
            vectorstore = PineconeVectorStore(index=pinecone_index, embedding=embeddings_model)
            vectorstore.add_documents(legal_docs)
            print("Upsert complete.")

        else:
            print("Processing Excel file...")
            excel_docs = await load_file(local_path)
            print(f"Created {len(excel_docs)} document chunks from Excel.")
            
            pinecone_docs = []
            for i, doc in enumerate(excel_docs):
                doc_metadata = {
                    **metadata,
                    "source": f"gs://{bucket_name}/{file_name}",
                    "chunk_id": i,
                    "sheet": doc.metadata.get("sheet", "Unknown"),
                    "chunk_index": doc.metadata.get("chunk_index", 0),
                    "total_chunks": len(excel_docs)
                }
                
                chunk_doc = Document(
                    page_content=doc.page_content,
                    metadata=doc_metadata
                )
                pinecone_docs.append(chunk_doc)

            print(f"Upserting {len(excel_docs)} documents to Pinecone index '{PINECONE_INDEX_NAME}'...")
            vectorstore = PineconeVectorStore(index=pinecone_index, embedding=embeddings_model)
            vectorstore.add_documents(pinecone_docs)
            print("Upsert complete.")

    except Exception as e:
        print(f"An error occurred during processing: {e}")
        raise
    finally:
        if os.path.exists(local_path):
            try:
                import time
                time.sleep(0.1)
                os.remove(local_path)
                print(f"Cleaned up temporary file: {local_path}")
            except Exception as cleanup_error:
                print(f"Warning: Error during cleanup of {local_path}: {cleanup_error}")

    print(f"--- Finished processing for {file_name} ---")

    return True 
