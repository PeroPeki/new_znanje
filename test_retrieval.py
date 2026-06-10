from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={"device": "cpu"},
)
db = Chroma(persist_directory="chroma_db/", embedding_function=embeddings)
retriever = db.as_retriever(search_kwargs={"k": 3})

pitanja = [
    "Koliki je doseg VW ID.3 Pro?",
    "Koji motori su dostupni za Golf?",
    "Kolika je garancija na bateriju ID.4?",
]

for p in pitanja:
    print(f"\nPitanje: {p}")
    rezultati = retriever.invoke(p)
    for r in rezultati:
        print(f"  [{r.metadata.get('source_file', '?')}]")
        print(f"  {r.page_content[:200]}")
        print()