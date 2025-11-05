import os
import logging
import json
from typing import List, Dict
from dotenv import load_dotenv
load_dotenv()

# ------------------------------
# Logging & LangSmith Tracing
# ------------------------------
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "rag-agent"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ------------------------------
# Imports
# ------------------------------
from src.rag_utils import build_or_load_vectorstore, retrieve_context

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
# ------------------------------
# Plan Node
# ------------------------------
def plan_node(query: str) -> Dict:
    """Interpret the query and decide if retrieval is needed."""
    logger.info("[plan] Interpreting query: %s", query)
    lowered = query.lower()
    retrieve_needed = False
    reasons = []

    if len(query) > 40:
        retrieve_needed = True
        reasons.append("query is long, likely needs context")

    if any(w in lowered for w in ["who", "what", "when", "where", "how", "does", "is"]):
        retrieve_needed = True
        reasons.append("query is factual or interrogative")

    logger.info("[plan] retrieve_needed=%s; reasons=%s", retrieve_needed, reasons)
    return {"query": query, "retrieve": retrieve_needed, "reasons": reasons}

# ------------------------------
# Retrieve Node
# ------------------------------
def retrieve_node(query: str, vectorstore, k: int = 3) -> List[str]:
    logger.info("[retrieve] Fetching top-%d contexts for query", k)
    contexts = retrieve_context(query, vectorstore, k=k)
    logger.info("[retrieve] Retrieved %d context chunks", len(contexts))
    return contexts

# ------------------------------
# Answer Node
# ------------------------------
def answer_node(query: str, contexts: List[str]) -> Dict:
    """
    Generate an answer using LLM given the query and context chunks.
    Returns a dict: {"answer": str, "source_chunks": List[str]}
    """
    logger.info("[answer] Building prompt and calling LLM")

    system_prompt = """
You are a helpful assistant that answers questions about provided context.
The context consists of chunks of documents.
Always respond in strict JSON format as:

{
  "answer": "<your concise answer here>",
  "source_chunks": ["chunk1 text used", "chunk2 text used"]
}

Do NOT include markdown, backticks, or extra characters outside JSON.
"""
    joined_context = "\n\n---\n\n".join(contexts)
    user_prompt = f"Context:\n{joined_context}\n\nQuestion: {query}"

    if ChatOpenAI is None:
        logger.warning("[answer] ChatOpenAI not available; returning stub answer.")
        return {"answer": "[LLM not configured]", "source_chunks": []}

    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    resp = llm.invoke(messages)
    raw_text = getattr(resp, "content", str(resp))
    logger.info("[answer] LLM response length: %d", len(raw_text))

    # Parse JSON safely
    try:
        answer_json = json.loads(raw_text)
    except json.JSONDecodeError:
        logger.warning("[answer] LLM output not valid JSON, falling back to raw text.")
        answer_json = {"answer": raw_text, "source_chunks": []}

    # Clean line breaks and extra spaces
    answer_json["answer"] = answer_json.get("answer", "").replace("\n", " ").replace("  ", " ").strip()
    if "source_chunks" not in answer_json or not isinstance(answer_json["source_chunks"], list):
        answer_json["source_chunks"] = []

    return answer_json

# ------------------------------
# Reflect Node
# ------------------------------
def reflect_node(query: str, answer_obj: Dict, contexts: List[str]) -> Dict:
    """
    Check if answer is relevant and complete.
    """
    logger.info("[reflect] Checking answer relevance & completeness")
    answer_text = answer_obj.get("answer", "")
    ok = True
    notes = []

    if len(answer_text) < 20:
        ok = False
        notes.append("Answer too short")

    if not answer_obj.get("source_chunks"):
        notes.append("Answer did not cite context chunks")

    result = {"ok": ok, "notes": notes, "answer": answer_text, "source_chunks": answer_obj.get("source_chunks", [])}
    logger.info("[reflect] ok=%s notes=%s", ok, notes)
    return result

# ------------------------------
# Workflow Runner
# ------------------------------
def run_agent(query: str, data_dir: str = "data", persist_dir: str = "chroma_db"):
    """
    Full RAG workflow: load vectorstore, retrieve context, answer, reflect
    """
    vectorstore = build_or_load_vectorstore(data_dir, persist_directory=persist_dir)

    plan = plan_node(query)
    contexts = []
    if plan["retrieve"]:
        contexts = retrieve_node(query, vectorstore, k=3)
    else:
        logger.info("[run_agent] Retrieval not required; proceeding without external context")

    answer_obj = answer_node(query, contexts)
    reflection = reflect_node(query, answer_obj, contexts)

    print("\n===== FINAL ANSWER =====\n")
    print(answer_obj["answer"])
    print("\n===== REFLECTION =====\n")
    print(reflection)

    return answer_obj, reflection

# ------------------------------
# CLI Entry
# ------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the RAG agent against actor PDFs.")
    parser.add_argument("--query", default=None, help="Question to ask the agent")
    args = parser.parse_args()

    if args.query is None:
        q = input("Enter a question: ").strip()
    else:
        q = args.query

    run_agent(q)
