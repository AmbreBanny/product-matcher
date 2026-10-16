# ÉTAPE 6 - EXPORT : préparer les résultats pour la démo web
import sqlite3, json, shutil
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

ALPHA = 0.6  # le meilleur équilibre trouvé à l'étape 5

con = sqlite3.connect("data/catalogue.db")
produits = pd.read_sql("SELECT * FROM produits", con)
emb = pd.read_sql("SELECT * FROM embeddings", con)
con.close()

vecteurs = {row.id: np.frombuffer(row.vecteur, dtype=np.float32) for row in emb.itertuples()}
A = produits[produits["catalogue"] == "A"].reset_index(drop=True)
B = produits[produits["catalogue"] == "B"].reset_index(drop=True)

# Score hybride image + texte
scores_image = np.stack([vecteurs[i] for i in A["id"]]) @ np.stack([vecteurs[i] for i in B["id"]]).T
model = SentenceTransformer("clip-ViT-B-32")
scores_texte = model.encode(A["nom"].tolist(), normalize_embeddings=True) @ \
               model.encode(B["nom"].tolist(), normalize_embeddings=True).T
scores = ALPHA * scores_image + (1 - ALPHA) * scores_texte

# Les 3 meilleurs équivalents pour chaque produit A
top3 = np.argsort(-scores, axis=1)[:, :3]

resultats = []
for i, a in A.iterrows():
    equivalents = []
    for j in top3[i]:
        b = B.iloc[j]
        equivalents.append({
            "id": int(b["id"]), "nom": b["nom"], "categorie": b["categorie"],
            "score": round(float(scores[i, j]), 3),
            "score_image": round(float(scores_image[i, j]), 3),
            "score_texte": round(float(scores_texte[i, j]), 3),
            "image": f"images/{b['id']}.jpg",
        })
    resultats.append({
        "id": int(a["id"]), "nom": a["nom"], "categorie": a["categorie"],
        "image": f"images/{a['id']}.jpg", "equivalents": equivalents,
    })

# On prépare le dossier du site : données + images
sortie = Path("docs")
(sortie / "images").mkdir(parents=True, exist_ok=True)
for chemin in produits["image"]:
    shutil.copy(chemin, sortie / "images")
with open(sortie / "data.json", "w", encoding="utf-8") as f:
    json.dump(resultats, f, ensure_ascii=False)

print(f"{len(resultats)} produits exportés dans docs/data.json")