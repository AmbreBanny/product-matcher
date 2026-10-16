# ÉTAPE 2 - EMBEDDINGS + STOCKAGE : calculer les empreintes une fois et les ranger en base
import sqlite3
import numpy as np
import pandas as pd
from PIL import Image
from sentence_transformers import SentenceTransformer

# On relit les données propres de l'étape 1
df = pd.read_csv("data/produits.csv")

# On simule deux fabricants : moitié A, moitié B
df["catalogue"] = np.where(np.arange(len(df)) < len(df) // 2, "A", "B")

# Calcul des empreintes (normalisées pour que la comparaison soit plus simple ensuite)
model = SentenceTransformer("clip-ViT-B-32")
images = [Image.open(chemin) for chemin in df["image"]]
embeddings = model.encode(images, batch_size=32, show_progress_bar=True, normalize_embeddings=True)

# Stockage dans une base SQLite : une table produits, une table embeddings
con = sqlite3.connect("data/catalogue.db")
df.to_sql("produits", con, if_exists="replace", index=False)

con.execute("DROP TABLE IF EXISTS embeddings")
con.execute("CREATE TABLE embeddings (id INTEGER PRIMARY KEY, vecteur BLOB)")
con.executemany(
    "INSERT INTO embeddings VALUES (?, ?)",
    [(int(i), v.astype(np.float32).tobytes()) for i, v in zip(df["id"], embeddings)],
)
con.commit()
con.close()

print(f"{len(df)} produits et leurs empreintes enregistrés dans data/catalogue.db")
print(df["catalogue"].value_counts())