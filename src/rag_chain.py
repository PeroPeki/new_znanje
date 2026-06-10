"""
rag_chain.py — Standardni RAG (LCEL implementacija)
Tip: Inference-phase RAG — dohvat relevantnih chunkova pri svakom upitu
Retrieval: Hibridni BM25 (keyword) + MMR semantički, EnsembleRetriever s RRF
"""

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from src.llm_loader import load_llm

CHROMA_DIR = "chroma_db/"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Mistral instruct format — bez <s> jer ga llama_cpp dodaje automatski
VW_PROMPT_TEMPLATE = """[INST] Odgovori na pitanje koristeći SAMO sljedeći kontekst. Kontekst sadrži tehničke podatke o Volkswagen vozilima. Uvijek odgovaraj na hrvatskom jeziku.

Pravila:
- Navedi SVE relevantne vrijednosti iz konteksta (npr. i osnovnu i maksimalnu vrijednost).
- Ne dodaj prefiks poput "Odgovor:" — odmah daj odgovor.
- Ako informacija nije u kontekstu, reci točno: "Nemam tu informaciju."
- Nemoj pretpostavljati ni izmišljati vrijednosti koje nisu u kontekstu.

Kontekst:
{context}

Pitanje: {question} [/INST]"""

# Model -> relevantne source datoteke (za ciljano pretraživanje)
MODEL_SOURCES = {
    "id.3": ["vw_id3_hr.txt", "ID3_MY26.pdf"],
    "id3":  ["vw_id3_hr.txt", "ID3_MY26.pdf"],
    "id.4": ["vw_id4_hr.txt", "EN_The_ID_4.pdf"],
    "id4":  ["vw_id4_hr.txt", "EN_The_ID_4.pdf"],
    "golf": ["vw_golf_hr.txt", "Golf_MY25.pdf", "37717-GolfPressPackJuly2024.pdf", "VW-Golf-GTI-Spec-Sheet_5 MARCH.pdf"],
    "tiguan": ["vw_tiguan_hr.txt", "VW-Spec-sheet-update-Tiguan-2025.pdf"],
    "tayron": ["vw_tayron_hr.txt", "75930 VW Tayron Specification Sheet 2025_SPEC_27 AUG (1).pdf"],
    "tcross": ["vw_tcross_hr.txt", "VW-T-Cross-Spec-sheet.pdf"],
    "t-cross": ["vw_tcross_hr.txt", "VW-T-Cross-Spec-sheet.pdf"],
    "polo": ["VW-POLO-EXRESS-SPEC-27-FEB.pdf", "POLO-SEDAN-SPEC-SHEET-2026.pdf", "81595 VW Polo Vivo Spec-sheet 2026_12 MAY.pdf"],
    "taigo": ["taigo_specsheet.pdf"],
    "passat": ["37031-NewVWPassatpresspackApril2024.pdf"],
    "touareg": ["VW-Brochure-Touareg-2024-Nov-spec-sheet.pdf"],
}


def _detect_source_filter(query: str):
    """Vraća popis source_file za prepoznati model u upitu."""
    q = query.lower()
    for keyword, sources in MODEL_SOURCES.items():
        if keyword in q:
            return sources
    return None


def load_rag_chain(retriever_k: int = 5):
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )

    # BM25 retriever — direktno podudaranje ključnih riječi (k=20 jer RRF treba širu listu)
    all_data = vectorstore.get()
    from langchain_core.documents import Document
    bm25_docs = [
        Document(page_content=pc, metadata={"source_file": meta.get("source_file", "")})
        for pc, meta in zip(all_data["documents"], all_data["metadatas"])
    ]
    bm25_retriever = BM25Retriever.from_documents(bm25_docs, k=20)

    # Semantički retriever (MMR smanjuje redundanciju)
    semantic_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": retriever_k, "fetch_k": 30, "lambda_mult": 0.7},
    )

    # Ensemble: BM25 keyword (60%) + semantički (40%), RRF spajanje
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, semantic_retriever],
        weights=[0.6, 0.4],
    )

    def smart_retrieve(query: str):
        """Ciljano pretraživanje — filtrira po modelu ako se prepozna u upitu."""
        source_filter = _detect_source_filter(query)
        if source_filter:
            # Ciljano: pretraži samo relevantne izvore za prepoznati model
            targeted_retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": retriever_k + 3,
                    "fetch_k": 40,
                    "lambda_mult": 0.6,
                    "filter": {"source_file": {"$in": source_filter}},
                },
            )
            docs = targeted_retriever.invoke(query)
        else:
            # Opće pitanje: koristi ensemble hibridni retriever
            docs = ensemble_retriever.invoke(query)
        # Ograniči kontekst na prvih retriever_k + 2 chunova
        return docs[: retriever_k + 2]

    retriever_runnable = RunnableLambda(smart_retrieve)

    prompt = PromptTemplate(
        template=VW_PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    llm = load_llm()

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {
            "context": retriever_runnable | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever_runnable


def query(chain_tuple, pitanje: str) -> dict:
    chain, retriever = chain_tuple

    odgovor = chain.invoke(pitanje)

    docs = retriever.invoke(pitanje)
    izvori = list(set(
        doc.metadata.get("source_file", "nepoznato") for doc in docs
    ))

    return {"odgovor": odgovor, "izvori": izvori}
