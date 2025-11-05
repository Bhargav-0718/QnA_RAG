import os
from typing import List
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from src.data_loader import load_pdfs_as_texts, chunk_documents
from dotenv import load_dotenv
load_dotenv()

def build_or_load_vectorstore(
    data_dir: str = "../data",
    persist_directory: str = "../chroma_db",
):
    """
    Build or load a Chroma vectorstore from multiple PDFs.
    """
    persist_directory = os.path.abspath(persist_directory)

    # Try loading existing Chroma
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        print(f"📦 Loading existing ChromaDB from: {persist_directory}")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        return Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    print(f"🧠 Creating new ChromaDB at: {persist_directory}")
    docs = load_pdfs_as_texts(data_dir)
    chunks = chunk_documents(docs)

    # Prepare data for Chroma ingestion
    texts = [c for c in chunks]
    metadatas = [{"source": "actor_wiki"} for _ in chunks]
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    chroma = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=persist_directory,
    )

    print(f"✅ Vectorstore built and persisted at: {persist_directory}")
    return chroma

def retrieve_context(query: str, vectorstore: Chroma, k: int = 3) -> List[str]:
    """
    Retrieve top-k context chunks relevant to a query.
    """
    results = vectorstore.similarity_search(query, k=k)
    contexts = []
    for doc in results:
        txt = getattr(doc, "page_content", None) or getattr(doc, "content", None) or str(doc)
        contexts.append(txt)
    return contexts
