# 🎬 Actor Wiki Q&A Agent

This project presents a **Retrieval-Augmented Generation (RAG) agent** for answering questions about actors using their wiki PDFs. By leveraging vector embeddings, ChromaDB, and a large language model (LLM), the system retrieves relevant information from documents and generates concise answers.

---

## 🚀 Project Overview

Manual information retrieval from multiple actor wiki PDFs can be time-consuming and error-prone. This project automates the process by:

- Loading actor wiki PDFs from the `data/` folder.
- Extracting and chunking text from PDFs for semantic retrieval.
- Creating embeddings using OpenAI and storing them in ChromaDB.
- Retrieving relevant chunks for a user query.
- Generating answers using an LLM based on retrieved context.
- Reflecting on answer quality and citing source chunks.

The end goal is to provide a **fast, interactive Q&A system** about actors with clear provenance for the generated answers.

---

## 🧩 Problem Statement

Challenges in actor-related Q&A include:

- Multiple sources with inconsistent formats (different wiki PDFs).
- Difficulty for users to manually extract accurate information.
- Ensuring generated answers are concise, relevant, and cite sources.
- Implementing evaluation and scoring is challenging without reference answers.

This project addresses these issues using a RAG approach to retrieve and generate answers efficiently.

---

## ⚙️ Workflow

1. **Build / Load Vectorstore**  
   PDFs are converted to text, chunked, embedded using OpenAI embeddings, and stored in ChromaDB for semantic search.

2. **Query Planning**  
   The agent determines whether a query requires retrieval from the vectorstore based on keywords, question type, and length.

3. **Context Retrieval**  
   Top-k relevant chunks are fetched from ChromaDB using semantic similarity search.

4. **Answer Generation**  
   The LLM (GPT-4o-mini) generates answers using the retrieved context. The output includes the answer and optionally cited source chunks.

5. **Reflection**  
   Evaluates answer completeness and relevance. Provides metadata such as whether context chunks were used.

6. **Evaluation (Optional)**  
   LLM evaluation was considered, but skipped in this version due to the lack of reference answers.
