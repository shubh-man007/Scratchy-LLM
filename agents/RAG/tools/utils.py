import logging
import re

from google.adk.tools.tool_context import ToolContext
from vertexai import rag

from rag_agent.config import (
    LOCATION,
    PROJECT_ID,
)

logger = logging.getLogger(__name__)


def get_corpus_resource_name(corpus_name: str) -> str:
    """
    Convert a corpus name to its full resource name if needed.
    Handles various input formats and ensures the returned name follows Vertex AI's requirements.

    Args:
        corpus_name (str): The corpus name or display name

    Returns:
        str: The full resource name of the corpus
    """
    logger.info(f"Getting resource name for corpus: {corpus_name}")

    if re.match(r"^projects/[^/]+/locations/[^/]+/ragCorpora/[^/]+$", corpus_name):
        return corpus_name

    try:
        corpora = rag.list_corpora()
        for corpus in corpora:
            if hasattr(corpus, "display_name") and corpus.display_name == corpus_name:
                return corpus.name
    except Exception as e:
        logger.warning(f"Error when checking for corpus display name: {str(e)}")
        pass

    if "/" in corpus_name:
        corpus_id = corpus_name.split("/")[-1]
    else:
        corpus_id = corpus_name

    corpus_id = re.sub(r"[^a-zA-Z0-9_-]", "_", corpus_id)
    return f"projects/{PROJECT_ID}/locations/{LOCATION}/ragCorpora/{corpus_id}"


def check_corpus_exists(corpus_name: str, tool_context: ToolContext) -> bool:
    """
    Check if a corpus with the given name exists.

    Args:
        corpus_name (str): The name of the corpus to check
        tool_context (ToolContext): The tool context for state management

    Returns:
        bool: True if the corpus exists, False otherwise
    """
    if tool_context.state.get(f"corpus_exists_{corpus_name}"):
        return True

    try:
        corpus_resource_name = get_corpus_resource_name(corpus_name)
        corpora = rag.list_corpora()
        for corpus in corpora:
            if (corpus.name == corpus_resource_name or corpus.display_name == corpus_name):
                tool_context.state[f"corpus_exists_{corpus_name}"] = True
                if not tool_context.state.get("current_corpus"):
                    tool_context.state["current_corpus"] = corpus_name
                return True
            
        return False
    except Exception as e:
        logger.error(f"Error checking if corpus exists: {str(e)}")
        return False


def set_current_corpus(corpus_name: str, tool_context: ToolContext) -> bool:
    """
    Set the current corpus in the tool context state.

    Args:
        corpus_name (str): The name of the corpus to set as current
        tool_context (ToolContext): The tool context for state management

    Returns:
        bool: True if the corpus exists and was set as current, False otherwise
    """
    if check_corpus_exists(corpus_name, tool_context):
        tool_context.state["current_corpus"] = corpus_name
        return True
    return False
