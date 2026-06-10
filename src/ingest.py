"""
ingest.py — Učitavanje dokumenata i izgradnja vektorske baze
RAG tip: Standardni RAG (inference-phase retrieval)
"""

import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DOCS_DIR = "documents/"
CHROMA_DIR = "chroma_db/"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def load_documents(directory: str):
    """Učitava sve podržane dokumente iz direktorija."""
    documents = []
    loaders = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".html": UnstructuredHTMLLoader,
    }
    for filename in os.listdir(directory):
        ext = os.path.splitext(filename)[1].lower()
        if ext in loaders:
            filepath = os.path.join(directory, filename)
            try:
                loader = loaders[ext](filepath)
                docs = loader.load()
                # Dodaj metadata (izvor dokumenta)
                for doc in docs:
                    doc.metadata["source_file"] = filename
                documents.extend(docs)
                print(f"✓ Učitan: {filename} ({len(docs)} stranica/dijelova)")
            except Exception as e:
                print(f"✗ Greška pri učitavanju {filename}: {e}")
    return documents

def chunk_documents(documents, chunk_size=300, chunk_overlap=75):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    # Dodaj prefiks svakom chunku — svi chunkovi dobivaju kontekst o izvoru
    for chunk in chunks:
        source = chunk.metadata.get("source_file", "")
        if source:
            chunk.page_content = f"Dokument: {source}\n\n{chunk.page_content}"
    print(f"\nUkupno chunkova: {len(chunks)}")
    return chunks

def build_vectorstore(chunks):
    """Generira embeddinge i pohranjuje ih u ChromaDB."""
    print(f"\nUčitavanje embedding modela: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print("Gradnja vektorske baze... (može potrajati nekoliko minuta)")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    print(f"✓ Vektorska baza izgrađena i pohranjena u: {CHROMA_DIR}")
    return vectorstore

if __name__ == "__main__":
    print("=== VW Chatbot — Indeksiranje dokumenata ===\n")
    docs = load_documents(DOCS_DIR)
    print(f"\nUkupno učitano dokumenata: {len(docs)}")
    chunks = chunk_documents(docs)
    vectorstore = build_vectorstore(chunks)
    print("\n✓ Indeksiranje završeno!")