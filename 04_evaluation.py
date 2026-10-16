# ÉTAPE 4 - ÉVALUATION : mesurer la fiabilité au lieu de juger à l'œil
import sqlite3
import pandas as pd

con = sqlite3.connect("data/catalogue.db")
m = pd.read_sql("SELECT * FROM matches", con)
produits = pd.read_sql("SELECT * FROM produits", con)
con.close()

# Un match est "correct" si les deux produits sont de la même catégorie
m["correct"] = m["categorie_A"] == m["categorie_B"]

# Un match correct n'est "possible" que si la catégorie existe dans le catalogue B
categories_B = set(produits.loc[produits["catalogue"] == "B", "categorie"])
m["possible"] = m["categorie_A"].isin(categories_B)

print(f"Précision globale : {m['correct'].mean():.0%}")
print(f"Précision quand un équivalent existe : {m.loc[m['possible'], 'correct'].mean():.0%}")
print(f"Produits A sans aucun équivalent possible dans B : {(~m['possible']).sum()}")

# Le score prédit-il la justesse ? On découpe les matchs en 4 tranches de score
m["tranche"] = pd.qcut(m["score"], 4, labels=["scores faibles", "moyens-", "moyens+", "scores forts"])
print("\nPrécision par tranche de score :")
print(m.groupby("tranche", observed=True)["correct"].mean().map("{:.0%}".format))

# Les erreurs les plus "sûres d'elles" : les plus dangereuses
print("\nErreurs avec le score le plus élevé :")
print(m[~m["correct"]].nlargest(5, "score")[["nom_A", "nom_B", "score"]].to_string(index=False))