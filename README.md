# Product Matcher

**Trouver automatiquement l'équivalent d'un produit dans un autre catalogue, en combinant l'analyse de l'image et du texte.**

🔗 **[Voir la démo en ligne](https://ambrebanny.github.io/product-matcher/)**

---

## Le problème

Une entreprise reçoit les catalogues de plusieurs fabricants et doit identifier, pour chaque produit, son équivalent chez un concurrent. Fait à la main, ce travail prend des jours et dépend de l'œil de la personne qui le fait.

Ce projet est né d'un besoin réel rencontré dans mon travail (équivalences entre catalogues d'équipements d'aires de jeux). Il a été validé ici sur un jeu de données public et étiqueté, pour pouvoir **mesurer** la fiabilité des résultats.

## La démarche : un pipeline de données en 7 étapes

| Étape | Script | Rôle |
|---|---|---|
| 1. Ingestion | `01_ingestion.py` | Télécharge 1 000 produits, range les images et les métadonnées (CSV) |
| 2. Embeddings | `02_embeddings.py` | Transforme chaque image en vecteur avec CLIP, stocke tout dans SQLite |
| 3. Matching | `03_matching.py` | Compare les 500 produits du catalogue A aux 500 du catalogue B (similarité cosinus) |
| 4. Évaluation | `04_evaluation.py` | Mesure la précision au lieu de juger à l'œil |
| 5. Amélioration | `05_hybride.py` | Combine score image et score texte, et cherche le meilleur équilibre |
| 6. Export | `06_export.py` | Prépare les résultats pour la démo web |
| 7. RAG vérifié | `07_rag.py` | Un LLM local explique les matchs, et chaque affirmation est vérifiée par le code |

## Les résultats

| Approche | Précision |
|---|---|
| Image seule (baseline) | 67 % |
| Texte seul | 70 % |
| **Hybride (60 % image + 40 % texte)** | **77 %** |
| Hybride + seuil de confiance | **83 %** (sur 75 % des produits) |

**Ce que j'en retiens :**
- **Les scores bruts sont trompeurs.** Mon premier test donnait 0,86 pour deux chaises et 0,85 pour une chaise et une tasse. Ce qui compte, c'est le classement des résultats, pas la valeur absolue du score.
- **L'image et le texte se complètent.** Combinés, ils font mieux que chacun séparément (+10 points par rapport à l'image seule).
- **Mieux vaut dire « je ne sais pas » que se tromper.** Avec un seuil de confiance, le système écarte 25 % des cas incertains et gagne 6 points de précision sur les autres.

## Le RAG et la lutte contre les hallucinations

Un modèle de langage local (Gemma 3 via Ollama) explique pourquoi deux produits correspondent, uniquement à partir de leurs fiches.

Un premier essai a montré que le modèle inventait des informations (par exemple un « usage estival » qui ne figurait nulle part). J'ai donc ajouté une **vérification par citation** : le modèle doit justifier chaque affirmation par une citation exacte des fiches, et le code rejette automatiquement toute affirmation dont la citation est introuvable.

Résultat : **19 affirmations vérifiées sur 21**, et plus aucune invention qui passe.

## Limites et pistes d'amélioration

- **Mesure un peu optimiste** : le poids image/texte a été choisi sur les mêmes données que celles évaluées. Il faudrait le valider sur un jeu de test séparé.
- **Évaluation sévère** : certaines « erreurs » sont des produits très proches (slip et boxer de la même gamme) classés dans des catégories voisines.
- **Le genre n'est pas visible à l'image** : des sandales homme peuvent matcher des ballerines femme. Un filtre par métadonnées réglerait ce cas.
- **Vérification RAG perfectible** : elle contrôle que la citation existe, mais pas qu'elle concerne le bon produit.
- **Prochaine étape** : appliquer la méthode à des catalogues réels d'équipements d'aires de jeux, avec un modèle plus puissant.

## Stack technique

Python · CLIP (sentence-transformers) · PyTorch · pandas · NumPy · SQLite · Ollama (Gemma 3) · HTML/CSS/JavaScript · GitHub Pages

## Lancer le projet

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

python 01_ingestion.py
python 02_embeddings.py
python 03_matching.py
python 04_evaluation.py
python 05_hybride.py
python 06_export.py
python 07_rag.py                # nécessite Ollama et le modèle gemma3:4b
```

## Données

Jeu de données public [Fashion Product Images (Small)](https://huggingface.co/datasets/ashraq/fashion-product-images-small), issu du dataset Kaggle de Param Aggarwal.

---

*Projet réalisé par Ambre Banny Lavilley dans le cadre de ma reconversion vers la data et l'IA.*