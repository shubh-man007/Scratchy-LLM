import os
from typing import List, Dict, Any
import re
import pandas as pd
import json
import markdown
import asyncio
import logging
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

try:
    llm_large = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
except Exception as e:
    logging.error(f"Failed to initialize LLM: {e}")
    raise

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]
)


logging.basicConfig(
    level=logging.INFO,  
    format='%(asctime)s %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p'
)


def clean_cell(val: Any) -> str:
    """Clean individual cell values."""
    if pd.isna(val):
        return ""
    val = str(val)
    val = re.sub(r'[ \t]+', ' ', val)
    val = val.replace(" Nan ", "")
    return val.strip()


def process_excel_structured(filepath: str) -> List[Dict]:
    all_chunks = []
    try:
        with pd.ExcelFile(filepath) as xlsx:
            for sheet_name in xlsx.sheet_names:
                df = xlsx.parse(sheet_name)
                df = df.dropna(how='all').dropna(axis=1, how='all')
                if df.empty:
                    continue

                headers = [clean_cell(col) for col in df.columns]
                rows = []
                for _, row in df.iterrows():
                    row_values = [clean_cell(val) for val in row]
                    if any(row_values):
                        rows.append(row_values)

                chunk = {
                    "type": "table",
                    "sheet": sheet_name,
                    "headers": headers,
                    "rows": rows
                }
                all_chunks.append(chunk)
        
    except Exception as e:
        logging.error(f"Error processing Excel file {filepath}: {e}")
        raise

    return all_chunks


def llm_prompt(sheet: dict, filename: str, sheet_name: str) -> str:
    return f"""
You are an expert data analyst. You are given a JSON object representing a single sheet from an Excel file named '{filename}', specifically the sheet '{sheet_name}'.

The JSON contains:
- "headers": a list of column headers (may include garbage like 'Unnamed: X' or empty strings)
- "rows": a list of rows, each a list of cell values (may include empty or irrelevant cells)

Your tasks:
1. **Carefully examine the headers and rows.**
2. **Discard any columns with headers that are empty, 'Unnamed', or clearly not useful.**
3. **Remove any rows that are empty or contain only irrelevant/empty data.**
4. **Extract and present only the meaningful, human-readable data.**
5. **If the sheet contains multiple tables or sections, separate them clearly.**
6. **Output the cleaned data as a well-formatted markdown table (or multiple tables if needed).**
7. **If there are important notes, instructions, or metadata, summarize them above the table in markdown.**
8. **Do not include any internal Excel metadata, garbage values, or empty indices.**

Here is the JSON for the sheet:
{sheet}

Think step by step, and only output clean, readable markdown. Make sure to return just the markdown table and nothing else.
"""


async def json_to_md_llm(excel_json: list[Dict], filename: str, llm, max_retries=3) -> tuple[list[Dict], int]:
    results = []
    sheet_count = 0
    for sheet in excel_json:
        sheet_name = sheet.get("sheet", "Unknown Sheet")
        prompt = llm_prompt(sheet, filename, sheet_name)

        for attempt in range(max_retries):
            try:
                logging.info(f"Processing {sheet_name}...")
                response = await llm.ainvoke(prompt)
                processed_content = {
                    "sheet": sheet_name,
                    "markdown": response.content
                }
                results.append(processed_content)
                break  
            except Exception as e:
                if "overloaded" in str(e).lower() and attempt < max_retries - 1:
                    logging.warning(f"LLM overloaded for {sheet_name}, retrying ({attempt+1}/{max_retries})...")
                    await asyncio.sleep(5) 
                else:
                    logging.error(f"LLM processing failed for {sheet_name}: {e}. Skipping.")
                    break
        sheet_count += 1
    return results, sheet_count


async def excel_to_document(filepath : str) -> tuple[List, int]:
    excel_chunk = process_excel_structured(filepath)
    res, sheet_count = await json_to_md_llm(excel_chunk, filepath, llm_large)
    res_chunks = []
    for i in range(len(res)):
        chunks = markdown_splitter.split_text(res[i]["markdown"])
        for j, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk.page_content,
                metadata={
                    "sheet": res[i]["sheet"],
                    "chunk_index": j,
                    "total_chunks_in_sheet": len(chunks),
                    "source": filepath
                }
            )
            res_chunks.append(doc)
    return res_chunks, sheet_count
