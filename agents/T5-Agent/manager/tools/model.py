import os
import sqlite3
from typing import Optional
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from google.adk.tools.tool_context import ToolContext

import warnings
warnings.filterwarnings("ignore")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# <------Initialize model weights and tokenizer------>
model_name = 't5-small'
model_path = os.path.join(os.path.dirname(__file__), "finetuned_model_2_epoch")
tokenizer = AutoTokenizer.from_pretrained(model_name)
finetuned_model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
finetuned_model = finetuned_model.to(device)

def generate_sql_query(
    context: str,
    question: str,
    tool_context: ToolContext
) -> dict:
    """
    Converts a natural language question into an SQL query using a fine-tuned T5 model.

    This tool reads the provided database schema and a user question, generates a corresponding
    SQL query using a fine-tuned T5 model, and optionally updates the agent's state with the
    question and result.

    Args:
        context (str): A string representation of the database schema or table/column descriptions.
                        Example: "CREATE TABLE table_name_11 (date VARCHAR, away_team VARCHAR)"
        question (str): A natural language question to convert into an SQL query.
                        Example: "On what Date did the Away team essendon play?"
        tool_context (ToolContext): The current agent tool context used for state tracking.

    Returns:
        dict: A dictionary containing the generated SQL query, input prompt, and status.
              Example:
              {
                  "status": "success",
                  "sql_query": "...",
                  "question": "...",
                  "context": "...",
                  "prompt": "..."
              }

    Example:
        INPUT PROMPT:
        Tables:
        CREATE TABLE table_name_11 (date VARCHAR, away_team VARCHAR)

        Question:
        On what Date did the Away team essendon play?

        Answer:

        FINE-TUNED MODEL - ZERO SHOT:
        SELECT date FROM table_name_11 WHERE away_team = "essendon"

    Instructions:
        You are a Text2SQL assistant integrated into an agent. Your job is to convert a user's natural language question into an executable SQL query based on the provided database schema. 
        The core logic is handled by a fine-tuned T5 model via the `generate_sql_query` tool. 
        Your responsibility is to ensure clean inputs, handle user mistakes gracefully, and validate outputs where necessary.

        ### Behaviors and Expectations:

        1. **Schema Format Validation**
        - The `context` should contain one or more `CREATE TABLE` statements using standard SQL syntax.
        - If the user provides only column names without types (e.g., `CREATE TABLE users (id, name)`), request clarification:
            - _"It looks like column types are missing. Could you update the schema with data types like `id INT`, `name VARCHAR`, etc.?"_

        2. **Missing or Mismatched Columns**
        - If a user's question mentions a column not found in the schema, respond with:
            - _"The column `X` isn't present in the schema. Please check the spelling or provide an updated schema."_

        3. **Ambiguity in Table or Column Naming**
        - If names include spaces, special characters, or are inconsistently capitalized, clarify:
            - _"Just to confirm, is the table name `User Info` or should it be `user_info`?"_

        4. **SQL Output Post-processing**
        - If the model omits a semicolon at the end, you may append one.
        - Ensure string values are properly quoted (`"value"` or `'value'` as needed).
        - The SQL query should only include columns and tables defined in the schema.

        5. **Return Format**
        - Your responses should always be a JSON dictionary with:
            - `status`: `"success"` or `"error"`
            - `sql_query`: string (only on success)
            - `message`: string (for error explanations or confirmation)
            
    """

    prompt = f"""Tables:
{context}

Question:
{question}

Answer:
"""

    try:
        inputs = tokenizer(prompt, return_tensors='pt').to(device)

        output = tokenizer.decode(
            finetuned_model.generate(
                inputs["input_ids"],
                max_new_tokens=200,
            )[0],
            skip_special_tokens=True
        )

        tool_context.state["last_sql_prompt"] = prompt
        tool_context.state["last_sql_query"] = output
        tool_context.state["last_sql_question"] = question

        return {
            "status": "success",
            "sql_query": output,
            "question": question,
            "context": context,
            "prompt": prompt
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to generate SQL query: {str(e)}"
        }


def review_database_schema(
    db_path: str,
    tool_context: ToolContext,
    table_name: Optional[str] = None
) -> dict:
    """
    Retrieves the database schema for either:
    - all tables in the database (default), or
    - a specific table (if `table_name` is provided).

    Args:
        db_path (str): Path to the SQLite database file.
        tool_context (ToolContext): Agent's tool context for state tracking.
        table_name (Optional[str]): Name of a specific table to review (default None).

    Returns:
        dict: Schema information.
              Example:
              {
                  "status": "success",
                  "schema": "CREATE TABLE ..."
              }
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        if table_name:
            cursor.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?;",
                (table_name,)
            )
            result = cursor.fetchone()
            if result is None:
                return {
                    "status": "error",
                    "message": f"Table '{table_name}' not found in database."
                }
            schema = result[0]
        else:
            cursor.execute(
                "SELECT name, sql FROM sqlite_master WHERE type='table';"
            )
            rows = cursor.fetchall()
            if not rows:
                return {
                    "status": "error",
                    "message": "No tables found in the database."
                }
            schema = "\n\n".join(
                [f"-- {name} --\n{sql}" for name, sql in rows if sql]
            )

        conn.close()
        tool_context.state["last_schema"] = schema

        return {
            "status": "success",
            "schema": schema
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to retrieve schema: {str(e)}"
        }


def list_tables(
    db_path: str,
    tool_context: ToolContext
) -> dict:
    """
    Lists all table names in the database.

    Args:
        db_path (str): Path to the SQLite database file.
        tool_context (ToolContext): Agent's tool context for state tracking.

    Returns:
        dict: Table list.
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()

        tool_context.state["last_table_list"] = tables

        return {
            "status": "success",
            "tables": tables
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to list tables: {str(e)}"
        }


def preview_table_data(
    db_path: str,
    table_name: str,
    limit: int,
    tool_context: ToolContext
) -> dict:
    """
    Retrieves the first few rows from a table.

    Args:
        db_path (str): Path to the SQLite database file.
        table_name (str): Name of the table to preview.
        limit (int): Number of rows to retrieve.
        tool_context (ToolContext): Agent's tool context for state tracking.

    Returns:
        dict: Preview rows.
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM {table_name} LIMIT ?;", (limit,))
        rows = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
        conn.close()

        preview = {
            "columns": columns,
            "rows": rows
        }
        tool_context.state["last_table_preview"] = preview

        return {
            "status": "success",
            "preview": preview
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to preview table '{table_name}': {str(e)}"
        }
