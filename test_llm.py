from src.llm_loader import load_llm

llm = load_llm()
odgovor = llm.invoke("Što je Volkswagen Golf?")
print(odgovor)