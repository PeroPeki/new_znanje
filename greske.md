# Greške i popravci — VW RAG Chatbot (Claude Code sesija)

## Popravljene greške

### 1. Krivi importi u svim datotekama

**Simptom:** `ImportError: cannot import name 'ModelProfile'` pri pokretanju

**Uzrok:** `langchain_huggingface` paket nije kompatibilan s Python 3.14.

**Popravljeno u:** `src/rag_chain.py`, `src/ingest.py`, `test_retrieval.py`, `visualize_embeddings.py`

| Staro | Novo |
|---|---|
| `from langchain_huggingface import HuggingFaceEmbeddings` | `from langchain_community.embeddings import HuggingFaceEmbeddings` |
| `from langchain_community.vectorstores import Chroma` | `from langchain_chroma import Chroma` |

---

### 2. Deprecated `vectorstore.persist()` poziv

**Simptom:** DeprecationWarning, u novijim verzijama langchain-chroma bacalo bi grešku

**Uzrok:** `langchain-chroma` 1.x automatski persista — ručni poziv je nepotreban i uklonjen.

**Popravljeno u:** `src/ingest.py:66`

---

### 3. Neslaganje verzija ChromaDB (kritično)

**Simptom:** `KeyError: '_type'` pri učitavanju postojeće ChromaDB baze

**Uzrok:** pip je downgradeao `chromadb` s `1.5.9` (kojim je baza izgrađena) na `0.6.3` (kojeg je zahtijevao stari `langchain-chroma 0.2.2`). Format baze nije bio kompatibilan.

**Rješenje:** Nadogradnja `langchain-chroma` na `1.1.0` koji podržava `chromadb 1.x`, te nadogradnja prateće `langchain` ekosustav stack na kompatibilne verzije:
- `langchain` → `1.3.4`
- `langchain-community` → `0.4.2`
- `langchain-text-splitters` → `1.1.2`
- `langchain-chroma` → `1.1.0`

---

### 4. `evaluation.py` koristio stari RetrievalQA API

**Simptom:** `TypeError` — `pokreni_evaluaciju` primala `chain` objekt i zvala `chain({"query": ...})`, ali nova LCEL implementacija vraća `(chain, retriever)` tuple

**Popravljeno u:** `src/evaluation.py` — prepisana funkcija `pokreni_evaluaciju` da prima `chain_tuple` i koristi `query()` funkciju iz `rag_chain.py`

Također dodano: `nltk.download("punkt_tab")` koji nedostajao i uzrokovao RuntimeError pri prvom pokretanju evaluacije.

---

### 5. `app.py` koristio krivi naziv varijable

**Simptom:** `TypeError: query() missing argument` — `query(chain, pitanje)` umjesto `query(chain_tuple, pitanje)`

**Popravljeno u:** `app.py` — `chain = init_chain()` → `chain_tuple = init_chain()`, i svaki poziv `query()` ažuriran

---

### 6. `visualize_embeddings.py` nije postojao

**Simptom:** `python visualize_embeddings.py` → `FileNotFoundError`

**Rješenje:** Datoteka kreirana s PCA i t-SNE vizualizacijom embeddinga, bojanje po izvoru dokumenta, pohrana u `embedding_vizualizacija.png`

---

### 7. Halucinacije — model ignorirao kontekst

**Simptom:** Model izmišljao informacije (Volvo, XC40...) umjesto da koristi RAG kontekst

**Uzrok:** Prompt nije bio u ispravnom Phi-3 instruct formatu (`<|system|>`, `<|user|>`, `<|assistant|>` tokeni). Model je primao "slobodni tekst" i ignorirao upute.

**Popravljeno u:** `src/rag_chain.py` — prompt preformuliran u Phi-3 chat template format

**Popravljeno u:** `src/llm_loader.py` — dodani stop tokeni (`<|end|>`, `<|user|>`, `<|system|>`) i `repeat_penalty=1.3`

---

### 8. Streamlit baca stotine `ModuleNotFoundError: torchvision` poruka

**Simptom:** Terminal prepun grešaka, teško čitati stvarne logove

**Uzrok:** Streamlit file watcher skenira sve module u `transformers` paketu, uključujući modele koji zahtijevaju `torchvision` (koji mi ne koristimo)

**Rješenje:** Kreiran `.streamlit/config.toml` s `fileWatcherType = "none"`

---

## Preostali problemi (nisu popravljeni)

### 9. Prompt leakage u odgovorima

**Simptom:** Model ispisuje dijelove system prompta u odgovor:
*"Nemam tu informaciju... Ne pretpostavljaj, ne izmišljirajte..."*

**Uzrok:** Phi-3-mini ponekad "zrcali" instrukcije iz system prompta nazad u output umjesto da ih primijeni

**Predloženo rješenje:**
- Skratiti i pojednostaviti system prompt — ukloniti negativne instrukcije
- Dodati `"Nemam"` u stop tokene kao sigurnosni net

---

### 10. Repetitivnost odgovora

**Simptom:** Model ponavlja isti paragraf dvaput u jednom odgovoru

**Uzrok:** `repeat_penalty=1.3` nije dovoljno visok za Phi-3-mini

**Predloženo rješenje:** Povećati `repeat_penalty` na `1.4`–`1.5`

---

### 11. Odgovori se sijeku na sredini rečenice

**Simptom:** Golf odgovor završava s *"ranu: Nemam tu informaciju"* — prekinut usred rečenice

**Uzrok:** `max_tokens=300` je prekratak za dulje, kompleksne odgovore

**Predloženo rješenje:** Povećati `max_tokens` na `512`

---

### 12. Loš hrvatski jezik u odgovorima

**Simptom:** Gramatički nesavršeni odgovori ("Volja VW Golf", "jednimo novi vrsta motora")

**Uzrok:** Phi-3-mini je primarno treniran na engleskom — kvaliteta na slavenskim jezicima je trajno ograničena veličinom modela

**Predloženo rješenje:** Zamjena modela — nije moguće popraviti prompt engineeringom

---

## Potencijalna poboljšanja

### Kratkoročno (bez zamjene modela)

| Poboljšanje | Datoteka | Prioritet |
|---|---|---|
| Popraviti prompt leakage (problemi 9–11) | `src/llm_loader.py`, `src/rag_chain.py` | Visok |
| MMR retrieval umjesto similarity — smanjuje duplikate u kontekstu | `src/rag_chain.py` | Srednji |
| Smanjiti `chunk_size` s 500 na 300 i ponovo izgraditi bazu — precizniji retrieval | `src/ingest.py` + ponovo pokrenuti `ingest.py` | Srednji |
| Fallback poruka ako retrieval ne pronađe ništa relevantno | `src/rag_chain.py` | Nizak |

### Dugoročno (zamjena modela)

Phi-3-mini-4k-instruct-q4 je premali za kvalitetne višejezične odgovore. Preporučeni modeli (svi dostupni kao .gguf za llama-cpp-python):

| Model | Veličina | Kvaliteta HR | RAM potreban |
|---|---|---|---|
| `Mistral-7B-Instruct-v0.3.Q4_K_M.gguf` | ~4.4 GB | Dobra | 8 GB |
| `Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf` | ~4.9 GB | Vrlo dobra | 8 GB |
| `gemma-2-9b-it-Q4_K_M.gguf` | ~5.5 GB | Izvrsna | 12 GB |

Promjena modela zahtijeva samo ažuriranje `MODEL_PATH` u `src/llm_loader.py` i prilagodbu prompt formata (svaki model ima drugačiji instruct template).
