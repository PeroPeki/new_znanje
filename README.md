# VW Chatbot — Lokalni RAG chatbot za Volkswagen vozila

Specijalizirani chatbot koji odgovara na pitanja o tehničkim specifikacijama Volkswagen vozila. Radi potpuno lokalno, bez internetske veze i bez vanjskih API-ja.

**Tehnologije:** Python · LangChain · ChromaDB · llama-cpp-python · Streamlit  
**RAG tip:** Inference-phase RAG s hibridnim retrieval-om (BM25 + MMR semantički, EnsembleRetriever)  
**Podržani jezici:** Hrvatski i engleski

---

## Preduvjeti

- Python 3.10 – 3.13
- ~6 GB slobodnog prostora na disku (za model)
- ~4 GB RAM-a (preporučeno 8 GB)

---

## Instalacija

### 1. Kloniraj repozitorij

```bash
git clone <url-repozitorija>
cd projekt_znanje
```

### 2. Kreiraj virtualno okruženje i instaliraj pakete

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Preuzmi model

Model se ne nalazi u repozitoriju zbog veličine — treba ga preuzeti ručno.

**Model:** `Mistral-7B-Instruct-v0.3-Q4_K_M.gguf`  
**Preuzimanje:** https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF

Preuzetu `.gguf` datoteku postavi u mapu `models/`:

```
projekt_znanje/
└── models/
    └── Mistral-7B-Instruct-v0.3-Q4_K_M.gguf
```

### 4. Izgradi vektorsku bazu

```bash
python -m src.ingest
```

Ovo učitava sve dokumente iz mape `documents/`, dijeli ih u chunkove, generira embedinge i pohranjuje ih u ChromaDB. Traje 2–5 minuta.

---

## Pokretanje

### Streamlit sučelje (preporučeno)

```bash
streamlit run app.py
```

Otvori http://localhost:8501 u pregledniku.

### Testiranje iz terminala

```bash
python test_rag.py
```

---

## Evaluacija

Pokreni automatsku evaluaciju s 20 pitanja (10 HR + 10 EN):

```bash
python -m src.evaluation
```

Rezultati se zapisuju u `evaluation_results.json`.

---

## Vizualizacija embeddinga

```bash
python visualize_embeddings.py
```

Generira `embedding_vizualizacija.png` s PCA i t-SNE prikazom embeddinga po dokumentu.

---

## Struktura projekta

```
projekt_znanje/
├── app.py                      # Streamlit sučelje
├── documents/                  # Izvorni dokumenti (PDF + TXT)
├── models/                     # LLM model (.gguf) — nije u repozitoriju
├── chroma_db/                  # Vektorska baza — generira se s ingest.py
├── src/
│   ├── ingest.py               # Učitavanje dokumenata i izgradnja baze
│   ├── rag_chain.py            # RAG lanac (retrieval + prompt + LLM)
│   ├── llm_loader.py           # Učitavanje lokalnog LLM modela
│   └── evaluation.py          # Evaluacija s metrikama
├── visualize_embeddings.py     # PCA + t-SNE vizualizacija embeddinga
├── evaluation_results.json     # Rezultati zadnje evaluacije
└── requirements.txt
```

---

## Dokumenti u bazi znanja

20 dokumenata (14 PDF + 6 TXT) koji pokrivaju modele:
ID.3, ID.4, Golf, Golf GTI, Tiguan, Tayron, T-Cross, Taigo, Polo, Passat, Touareg

---

## Napomena o hardveru

`n_gpu_layers=-1` u `llm_loader.py` znači da će model koristiti GPU ako je dostupan (CUDA/Metal). Na CPU-u generiranje odgovora traje 20–60 sekundi po pitanju, ovisno o procesoru.
