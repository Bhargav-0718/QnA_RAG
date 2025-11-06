import os
import json
import logging
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "rag-agent"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from src.data_loader import load_pdfs_as_texts, chunk_documents
from src.rag_utils import build_or_load_vectorstore, retrieve_context

from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage, SystemMessage

def plan_node(query: str) -> Dict:
    logger.info("[plan] Interpreting query: %s", query)
    lowered = query.lower()
    retrieve_needed = False
    reasons = []

    if "who" in lowered or "what" in lowered or "when" in lowered or "where" in lowered or "how" in lowered or "does" in lowered or "is" in lowered:
        retrieve_needed = True
        reasons.append("query is factual/interrogative")

    if len(query) > 40:
        retrieve_needed = True
        reasons.append("query is long, likely needs context")

    logger.info("[plan] retrieve_needed=%s; reasons=%s", retrieve_needed, reasons)
    return {"query": query, "retrieve": retrieve_needed, "reasons": reasons}

def retrieve_node(query: str, vectorstore, k: int = 3) -> List[str]:
    logger.info("[retrieve] Fetching top-%d contexts for query", k)
    contexts = retrieve_context(query, vectorstore, k=k)
    logger.info("[retrieve] Retrieved %d context chunks", len(contexts))
    return contexts

def answer_node(query: str, contexts: List[str]) -> Dict:
    logger.info("[answer] Building prompt and calling LLM")

    system_prompt = """
You are a helpful assistant that answers questions about provided context.
The context consists of document chunks.
Always respond in strict JSON format:

{
  "answer": "<your concise answer here>",
  "source_chunks": ["chunk1 text used", "chunk2 text used"]
}

Do NOT include markdown, backticks, or extra characters.
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

    try:
        answer_json = json.loads(raw_text)
    except json.JSONDecodeError:
        logger.warning("[answer] LLM output not valid JSON, using raw text fallback.")
        answer_json = {"answer": raw_text, "source_chunks": []}

    answer_json["answer"] = answer_json.get("answer", "").replace("\n", " ").strip()
    if "source_chunks" not in answer_json or not isinstance(answer_json["source_chunks"], list):
        answer_json["source_chunks"] = []

    return answer_json

def reflect_node(query: str, answer_obj: Dict, contexts: List[str]) -> Dict:
    logger.info("[reflect] Checking answer relevance & completeness")
    answer_text = answer_obj.get("answer", "")
    ok = True
    notes = []

    if len(answer_text) < 20:
        ok = False
        notes.append("Answer too short")
    if not answer_obj.get("source_chunks"):
        notes.append("Answer did not cite context chunks")

    return {"ok": ok, "notes": notes, "answer": answer_text, "source_chunks": answer_obj.get("source_chunks", [])}

def run_agent(query: str, vectorstore=None, data_dir: str = "../data", persist_dir: str = "../chroma_db", reference: str = None):
    if vectorstore is None:
        vectorstore = build_or_load_vectorstore(data_dir, persist_directory=persist_dir)

    plan = plan_node(query)
    contexts = retrieve_node(query, vectorstore) if plan["retrieve"] else []
    answer_obj = answer_node(query, contexts)
    reflection = reflect_node(query, answer_obj, contexts)

    return answer_obj, reflection


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG Agent with LLM-as-Judge")
    parser.add_argument("--query", default=None, help="Question to ask the agent")
    args = parser.parse_args()

    q = args.query or input("Enter your question: ").strip()
    run_agent(q)
