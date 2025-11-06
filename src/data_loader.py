import os
from typing import List
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_pdfs_as_texts(pdf_dir: str) -> List[str]:
    pdf_texts = []
    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            path = os.path.join(pdf_dir, filename)
            logger.info(f"Extracting text from {path}")
            try:
                reader = PdfReader(path)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""
                pdf_texts.append(text)
            except Exception as e:
                logger.error(f"Failed to read {filename}: {e}")
    return pdf_texts


def chunk_documents(texts: List[str], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    for text in texts:
        chunks.extend(splitter.split_text(text))
    return chunks
