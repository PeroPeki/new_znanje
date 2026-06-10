import json
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
import nltk
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

_sem_model = None

def _get_sem_model():
    global _sem_model
    if _sem_model is None:
        _sem_model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
    return _sem_model

TEST_PITANJA = [
    # --- HRVATSKI (10) ---
    {
        "jezik": "HR",
        "pitanje": "Koliki je WLTP doseg VW ID.3 Pro S?",
        "ocekivani_odgovor": "VW ID.3 Pro S ima WLTP kombinirani doseg od 548 do 564 km.",
    },
    {
        "jezik": "HR",
        "pitanje": "Kolika je kapacitet baterije VW ID.3 GTX?",
        "ocekivani_odgovor": "VW ID.3 GTX ima bateriju kapaciteta 79 kWh.",
    },
    {
        "jezik": "HR",
        "pitanje": "Koliki je električni doseg VW Golf eHybrid prema WLTP-u?",
        "ocekivani_odgovor": "VW Golf eHybrid ima električni doseg od 143 km prema WLTP mjerenju.",
    },
    {
        "jezik": "HR",
        "pitanje": "Koliki je prtljažnik VW Tiguana?",
        "ocekivani_odgovor": "Prtljažnik VW Tiguana iznosi 652 litre, a sa sklopljenim naslonima 1.650 litara.",
    },
    {
        "jezik": "HR",
        "pitanje": "Kolika je snaga VW Tayron R-Line motora?",
        "ocekivani_odgovor": "VW Tayron R-Line ima motor 1.4 TSI snage 110 kW ili 150 KS.",
    },
    {
        "jezik": "HR",
        "pitanje": "Kolika je garancija na bateriju VW ID.4?",
        "ocekivani_odgovor": "VW garantira najmanje 70% kapaciteta baterije VW ID.4 nakon 8 godina ili 160.000 km.",
    },
    {
        "jezik": "HR",
        "pitanje": "Koliki je prtljažnik VW ID.3?",
        "ocekivani_odgovor": "Prtljažnik VW ID.3 iznosi 385 litara, a sa sklopljenim stražnjim naslonima 1.267 litara.",
    },
    {
        "jezik": "HR",
        "pitanje": "Koji motor ima VW T-Cross R-Line?",
        "ocekivani_odgovor": "VW T-Cross R-Line ima motor 1.0 TSI snage 85 kW ili 116 KS s automatskim DSG mjenjačem.",
    },
    {
        "jezik": "HR",
        "pitanje": "Kolika je kombinirana potrošnja VW Tiguan 2.0 TDI?",
        "ocekivani_odgovor": "Kombinirana potrošnja VW Tiguan 2.0 TDI iznosi 6,6 litara na 100 km.",
    },
    {
        "jezik": "HR",
        "pitanje": "Kolika je maksimalna brzina VW ID.3 GTX?",
        "ocekivani_odgovor": "Maksimalna brzina VW ID.3 GTX je 180 km/h.",
    },
    # --- ENGLESKI (10) ---
    {
        "jezik": "EN",
        "pitanje": "What is the WLTP range of the VW ID.3 GTX?",
        "ocekivani_odgovor": "The VW ID.3 GTX has a WLTP combined range of 579 to 605 km.",
    },
    {
        "jezik": "EN",
        "pitanje": "What is the boot space of the VW ID.4?",
        "ocekivani_odgovor": "The VW ID.4 has a boot space of 543 litres, expanding to 1,575 litres with the rear seats folded.",
    },
    {
        "jezik": "EN",
        "pitanje": "What is the maximum DC charging speed of the VW ID.4?",
        "ocekivani_odgovor": "The VW ID.4 supports a maximum DC fast charging speed of 135 kW.",
    },
    {
        "jezik": "EN",
        "pitanje": "How long does it take to charge the VW ID.4 from 5% to 80%?",
        "ocekivani_odgovor": "The VW ID.4 charges from 5% to 80% in approximately 36 minutes using a 135 kW DC charger.",
    },
    {
        "jezik": "EN",
        "pitanje": "What engines are available for the VW T-Cross?",
        "ocekivani_odgovor": "The VW T-Cross is available with a 1.0 TSI 70 kW engine with manual gearbox and a 1.0 TSI 85 kW engine with DSG automatic gearbox.",
    },
    {
        "jezik": "EN",
        "pitanje": "What is the wheelbase of the VW Tiguan?",
        "ocekivani_odgovor": "The VW Tiguan has a wheelbase of 2,676 mm.",
    },
    {
        "jezik": "EN",
        "pitanje": "What is the towing capacity of the VW T-Cross?",
        "ocekivani_odgovor": "The VW T-Cross can tow up to 1,200 kg with a braked trailer.",
    },
    {
        "jezik": "EN",
        "pitanje": "What is the fuel consumption of the VW Golf 2.0 TDI 116 hp?",
        "ocekivani_odgovor": "The VW Golf 2.0 TDI 116 hp has a combined WLTP fuel consumption of 4.3 to 4.5 litres per 100 km.",
    },
    {
        "jezik": "EN",
        "pitanje": "What is the battery capacity of the VW Golf eHybrid?",
        "ocekivani_odgovor": "The VW Golf eHybrid has a lithium-ion battery with a capacity of 19.7 kWh.",
    },
    {
        "jezik": "EN",
        "pitanje": "What is the 0-100 km/h acceleration of the VW ID.4 Pro Performance?",
        "ocekivani_odgovor": "The VW ID.4 Pro Performance accelerates from 0 to 100 km/h in 8.5 seconds.",
    },
]


def izracunaj_rouge(hipoteza: str, referenca: str) -> dict:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    scores = scorer.score(referenca, hipoteza)
    return {
        "rouge1_f": round(scores["rouge1"].fmeasure, 4),
        "rouge2_f": round(scores["rouge2"].fmeasure, 4),
        "rougeL_f": round(scores["rougeL"].fmeasure, 4),
    }


def izracunaj_semanticku_slicnost(hipoteza: str, referenca: str) -> float:
    model = _get_sem_model()
    emb_h = model.encode(hipoteza, convert_to_tensor=True)
    emb_r = model.encode(referenca, convert_to_tensor=True)
    score = util.cos_sim(emb_h, emb_r).item()
    return round(score, 4)


def pokreni_evaluaciju(output_file: str = "evaluation_results.json"):
    from src.rag_chain import load_rag_chain, query

    rezultati = []

    for test in TEST_PITANJA:
        jezik = test.get("jezik", "HR")
        print(f"\n[{jezik}] {test['pitanje']}")

        chain_tuple = load_rag_chain(retriever_k=12)

        res = query(chain_tuple, test["pitanje"])
        chatbot_odgovor = res["odgovor"].strip()

        rouge = izracunaj_rouge(chatbot_odgovor, test["ocekivani_odgovor"])
        sem = izracunaj_semanticku_slicnost(chatbot_odgovor, test["ocekivani_odgovor"])

        rezultat = {
            "jezik": jezik,
            "pitanje": test["pitanje"],
            "ocekivani_odgovor": test["ocekivani_odgovor"],
            "chatbot_odgovor": chatbot_odgovor,
            "rouge_scores": rouge,
            "semantic_similarity": sem,
        }
        rezultati.append(rezultat)
        print(f"  Sem: {sem:.4f} | ROUGE-L: {rouge['rougeL_f']:.4f}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(rezultati, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Rezultati evaluacije pohranjeni u: {output_file}")
    return rezultati


if __name__ == "__main__":
    pokreni_evaluaciju()