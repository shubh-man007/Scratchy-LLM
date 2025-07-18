import os
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
