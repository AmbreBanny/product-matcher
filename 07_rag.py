# ÉTAPE 7 - RAG + VÉRIFICATION : l'IA explique, Python vérifie chaque affirmation
import json, random
import pandas as pd
import ollama

MODELE = "gemma3:4b"

# R - Récupération : les matchs déjà calculés + les couleurs des produits
with open("docs/data.json", encoding="utf-8") as f:
    data = json.load(f)
couleurs = pd.read_csv("data/produits.csv").set_index("id")["couleur"].to_dict()

def fiche(p):
    return (f"- Nom : {p['nom']}\n"
            f"- Catégorie : {p['categorie']}\n"
            f"- Couleur : {couleurs.get(p['id'], 'inconnue')}")

# A - Augmentation : consignes + fiches, avec obligation de citer
CONSIGNES = """Tu compares deux fiches produits.
Règles strictes :
- Utilise UNIQUEMENT les informations des fiches. N'invente rien.
- Chaque affirmation doit être accompagnée d'une citation COPIÉE MOT POUR MOT depuis une fiche.
- Si tu ne peux pas citer la fiche, n'écris pas l'affirmation.
- Réponds en français, en JSON, avec exactement ces clés :
  "verdict" : "équivalent", "partiellement équivalent" ou "pas équivalent"
  "constats" : liste d'objets {"affirmation": "...", "citation": "texte exact de la fiche"}
  "confiance" : "haute", "moyenne" ou "basse"
"""

total_ok, total_rejetes = 0, 0
random.seed(7)

for p in random.sample(data, 5):
    e = p["equivalents"][0]
    fiches = f"FICHE A (produit recherché)\n{fiche(p)}\n\nFICHE B (équivalent proposé)\n{fiche(e)}"

    # G - Génération : température 0, format JSON imposé
    reponse = ollama.chat(
        model=MODELE,
        messages=[{"role": "user", "content": f"{CONSIGNES}\n\n{fiches}"}],
        format="json",
        options={"temperature": 0},
    )
    explication = json.loads(reponse["message"]["content"])

    # VÉRIFICATION : chaque citation doit exister réellement dans les fiches
    texte_source = fiches.lower()
    print("=" * 70)
    print(f"A : {p['nom']}")
    print(f"B : {e['nom']}  (score {e['score']})")
    print(f"Verdict : {explication.get('verdict')}  |  Confiance : {explication.get('confiance')}\n")

    for c in explication.get("constats", []):
        citation = str(c.get("citation", "")).strip().lower()
        if citation and citation in texte_source:
            total_ok += 1
            print(f"  ✔ {c.get('affirmation')}   [« {c.get('citation')} »]")
        else:
            total_rejetes += 1
            print(f"  ✘ REJETÉ : {c.get('affirmation')}   (citation introuvable)")

print("=" * 70)
total = total_ok + total_rejetes
if total:
    print(f"Affirmations vérifiées : {total_ok}/{total}  |  rejetées : {total_rejetes} ({total_rejetes / total:.0%})")