import streamlit as st
import os
from agent import run_workflow

st.set_page_config(page_title="RAG Contract Q&A", layout="centered")
st.title("RAG Contract Q&A")

pdf_path = st.text_input("Path to contract PDF", value=os.path.join("data", "contract.pdf"))
query = st.text_input("Your question about the contract")

if st.button("Ask"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Running RAG agent (this may take a while on first run)..."):
            try:
                answer, reflection = run_workflow(query, pdf_path)
                st.subheader("Answer")
                st.write(answer)
                st.subheader("Reflection / metadata")
                st.json(reflection)
            except Exception as e:
                st.error(f"Error running agent: {e}")
