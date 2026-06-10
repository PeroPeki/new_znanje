import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

CHROMA_DIR = "chroma_db/"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def vizualiziraj_embeddings():
    embeddings_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings_model,
    )
    kolekcija = vectorstore._collection.get(include=["embeddings", "metadatas"])
    vektori = np.array(kolekcija["embeddings"])
    dokumenti = kolekcija["metadatas"]
    sources = [m.get("source_file", "unknown") for m in dokumenti]
    unique_sources = list(set(sources))
    boje = plt.cm.tab20(np.linspace(0, 1, len(unique_sources)))
    source_to_color = {s: boje[i] for i, s in enumerate(unique_sources)}

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    pca = PCA(n_components=2)
    pca_rezultat = pca.fit_transform(vektori)
    for source in unique_sources:
        mask = [s == source for s in sources]
        axes[0].scatter(
            pca_rezultat[mask, 0], pca_rezultat[mask, 1],
            c=[source_to_color[source]], label=source[:25], alpha=0.7, s=20,
        )
    axes[0].set_title("PCA vizualizacija embeddinga")
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(vektori) - 1))
    tsne_rezultat = tsne.fit_transform(vektori)
    for source in unique_sources:
        mask = [s == source for s in sources]
        axes[1].scatter(
            tsne_rezultat[mask, 0], tsne_rezultat[mask, 1],
            c=[source_to_color[source]], label=source[:25], alpha=0.7, s=20,
        )
    axes[1].set_title("t-SNE vizualizacija embeddinga")
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)

    plt.tight_layout()
    plt.savefig("embedding_vizualizacija.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✓ Vizualizacija pohranjena: embedding_vizualizacija.png")

if __name__ == "__main__":
    vizualiziraj_embeddings()
