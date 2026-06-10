import json
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

TEST_PITANJA = [
    {
        "pitanje": "Koliki je doseg VW ID.3 Pro S?",
        "ocekivani_odgovor": "VW ID.3 Pro S ima doseg do 564 km prema WLTP mjerenju.",
    },
    {
        "pitanje": "Koji motori su dostupni za VW Golf?",
        "ocekivani_odgovor": "Golf je dostupan s 1.5 TSI benzinskim, MHEV hibridnim, eHybrid plug-in hibridnim i 2.0 TDI dizelskim motorima.",
    },
    {
        "pitanje": "Kolika je garancija na bateriju VW ID.4?",
        "ocekivani_odgovor": "VW garantira najmanje 70% kapaciteta baterije nakon 8 godina ili 160.000 km.",
    },
    {
        "pitanje": "Koliki je prtljažnik VW ID.4?",
        "ocekivani_odgovor": "Prtljažnik VW ID.4 iznosi 543 litre, a sa sklopljenim naslonima 1.575 litara.",
    },
    {
        "pitanje": "Kolika je maksimalna snaga punjenja VW ID.3 GTX?",
        "ocekivani_odgovor": "ID.3 GTX podržava maksimalnu DC snagu punjenja od 185 kW.",
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


def izracunaj_bleu(hipoteza: str, referenca: str) -> float:
    smoother = SmoothingFunction().method1
    ref_tokens = [referenca.lower().split()]
    hyp_tokens = hipoteza.lower().split()
    score = sentence_bleu(ref_tokens, hyp_tokens, smoothing_function=smoother)
    return round(score, 4)


def pokreni_evaluaciju(output_file: str = "evaluation_results.json"):
    from src.rag_chain import load_rag_chain, query

    rezultati = []

    for test in TEST_PITANJA:
        print(f"\nPitanje: {test['pitanje']}")

        # Učitaj novi chain za svako pitanje — resetira KV cache i izbjegava overflow
        chain_tuple = load_rag_chain()

        res = query(chain_tuple, test["pitanje"])
        chatbot_odgovor = res["odgovor"].strip()

        rouge = izracunaj_rouge(chatbot_odgovor, test["ocekivani_odgovor"])
        bleu = izracunaj_bleu(chatbot_odgovor, test["ocekivani_odgovor"])

        rezultat = {
            "pitanje": test["pitanje"],
            "ocekivani_odgovor": test["ocekivani_odgovor"],
            "chatbot_odgovor": chatbot_odgovor,
            "rouge_scores": rouge,
            "bleu_score": bleu,
        }
        rezultati.append(rezultat)
        print(f"BLEU: {bleu} | ROUGE-L: {rouge['rougeL_f']}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(rezultati, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Rezultati evaluacije pohranjeni u: {output_file}")
    return rezultati


if __name__ == "__main__":
    pokreni_evaluaciju()