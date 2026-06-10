from src.rag_chain import load_rag_chain, query

print("Učitavanje RAG lanca...")
chain_tuple = load_rag_chain()

print("Postavljanje pitanja...")
rezultat = query(chain_tuple, "Koliki je doseg VW ID.3 Pro S?")
print("\nOdgovor:", rezultat["odgovor"])
print("Izvori:", rezultat["izvori"])