# ÉTAPE 3 - MATCHING : pour chaque produit A, trouver le plus ressemblant dans B
import sqlite3
import numpy as np
import pandas as pd

con = sqlite3.connect("data/catalogue.db")
produits = pd.read_sql("SELECT * FROM produits", con)
emb = pd.read_sql("SELECT * FROM embeddings", con)

# On reconvertit les empreintes stockées en listes de nombres
vecteurs = {row.id: np.frombuffer(row.vecteur, dtype=np.float32) for row in emb.itertuples()}

A = produits[produits["catalogue"] == "A"].reset_index(drop=True)
B = produits[produits["catalogue"] == "B"].reset_index(drop=True)
VA = np.stack([vecteurs[i] for i in A["id"]])
VB = np.stack([vecteurs[i] for i in B["id"]])

# Comparaison de tout A avec tout B en une opération (500 x 500 scores)
scores = VA @ VB.T
meilleur = scores.argmax(axis=1)  # pour chaque produit A, l'indice du meilleur B

matches = pd.DataFrame({
    "id_A": A["id"],
    "nom_A": A["nom"],
    "categorie_A": A["categorie"],
    "id_B": B["id"].values[meilleur],
    "nom_B": B["nom"].values[meilleur],
    "categorie_B": B["categorie"].values[meilleur],
    "score": scores[np.arange(len(A)), meilleur],
})

# On enregistre les résultats en base
matches.to_sql("matches", con, if_exists="replace", index=False)
con.close()

print(f"{len(matches)} matchs calculés\n")
print("Exemples :")
print(matches.sample(8, random_state=1)[["nom_A", "nom_B", "score"]].to_string(index=False))