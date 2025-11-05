import streamlit as st
from src.agent import run_agent
from src.rag_utils import build_or_load_vectorstore
from dotenv import load_dotenv
load_dotenv()

DATA_DIR = "data"        
CHROMA_DIR = "chroma_db" 

st.set_page_config(page_title="Actor Wiki Q&A", layout="centered")
st.title("🎬 Actor Wiki Q&A")
st.markdown(
    "Ask questions about the actors based on their wiki PDFs stored in the `data/` folder."
)

if "vectorstore" not in st.session_state:
    with st.spinner("Loading or building vectorstore..."):
        st.session_state["vectorstore"] = build_or_load_vectorstore(
            data_dir=DATA_DIR,
            persist_directory=CHROMA_DIR
        )
    st.success("✅ Vectorstore ready!")

vectorstore = st.session_state["vectorstore"]

query = st.text_input("Type your question here:")

if st.button("Ask"):
    if not query.strip():
        st.warning("Please enter a question!")
    else:
        with st.spinner("Running the agent..."):
            try:
                answer, reflection = run_agent(
                    query,
                    data_dir=DATA_DIR,
                    persist_dir=CHROMA_DIR
                )

                st.subheader("Answer")
                st.write(answer.get("answer", "[No answer returned]"))

                st.subheader("Reflection / Metadata")
                st.json(reflection)
            except Exception as e:
                st.error(f"Error running agent: {e}")

import os
pdf_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]
if pdf_files:
    st.info(f"Loaded PDFs: {', '.join(pdf_files)}")
else:
    st.warning(f"No PDFs found in `{DATA_DIR}/`. Please add actor wiki PDFs.")