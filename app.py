"""
app.py — Streamlit sučelje za VW Chatbot
Izborne komponente: GUI + Vizualizacija embeddinga
"""

import streamlit as st
from src.rag_chain import load_rag_chain, query

# Konfiguracija stranice
st.set_page_config(
    page_title="VW Chatbot 2026",
    page_icon="🚗",
    layout="centered",
)

st.title("🚗 Volkswagen Asistent 2026")
st.markdown("Postavite pitanja o VW vozilima — specifikacije, motori, oprema, servis.")

# Inicijalizacija RAG lanca (cachira se između poruka)
@st.cache_resource
def init_chain():
    with st.spinner("Učitavanje modela i baze znanja..."):
        return load_rag_chain()

chain_tuple = init_chain()

# Povijest razgovora
if "messages" not in st.session_state:
    st.session_state.messages = []

# Prikaz prethodnih poruka
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Unos pitanja
if pitanje := st.chat_input("Npr. Koje su tehničke specifikacije VW Golf 2026?"):
    # Prikaz korisničke poruke
    with st.chat_message("user"):
        st.markdown(pitanje)
    st.session_state.messages.append({"role": "user", "content": pitanje})

    # Generiranje odgovora
    with st.chat_message("assistant"):
        with st.spinner("Pretraživanje baze znanja..."):
            rezultat = query(chain_tuple, pitanje)

        st.markdown(rezultat["odgovor"])

        # Prikaz izvora
        if rezultat["izvori"]:
            with st.expander("📚 Izvorni dokumenti"):
                for izvor in rezultat["izvori"]:
                    st.markdown(f"- `{izvor}`")

        st.session_state.messages.append({
            "role": "assistant",
            "content": rezultat["odgovor"],
        })