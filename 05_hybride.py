# ÉTAPE 5 - AMÉLIORATION : combiner l'image ET le nom du produit
import sqlite3
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

con = sqlite3.connect("data/catalogue.db")
produits = pd.read_sql("SELECT * FROM produits", con)
emb = pd.read_sql("SELECT * FROM embeddings", con)
con.close()

vecteurs = {row.id: np.frombuffer(row.vecteur, dtype=np.float32) for row in emb.itertuples()}
A = produits[produits["catalogue"] == "A"].reset_index(drop=True)
B = produits[produits["catalogue"] == "B"].reset_index(drop=True)

# Score image (comme à l'étape 3)
scores_image = np.stack([vecteurs[i] for i in A["id"]]) @ np.stack([vecteurs[i] for i in B["id"]]).T

# Score texte : CLIP encode aussi les noms des produits
model = SentenceTransformer("clip-ViT-B-32")
TA = model.encode(A["nom"].tolist(), normalize_embeddings=True)
TB = model.encode(B["nom"].tolist(), normalize_embeddings=True)
scores_texte = TA @ TB.T

# On teste plusieurs équilibres entre image et texte
print("alpha = part de l'image dans le score final\n")
for alpha in [1.0, 0.8, 0.6, 0.5, 0.4, 0.2, 0.0]:
    scores = alpha * scores_image + (1 - alpha) * scores_texte
    meilleur = scores.argmax(axis=1)
    precision = (A["categorie"].values == B["categorie"].values[meilleur]).mean()
    print(f"alpha = {alpha:.1f}  →  précision {precision:.0%}")