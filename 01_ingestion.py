# ÉTAPE 1 - INGESTION : récupérer les données brutes et les ranger proprement
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image
from pathlib import Path
import pandas as pd
import io

N = 1000  # on travaille sur 1000 produits pour aller vite
REPO = "ashraq/fashion-product-images-small"

# On télécharge le fichier de données du catalogue (format Parquet)
fichiers = [f for f in HfApi().list_repo_files(REPO, repo_type="dataset") if f.endswith(".parquet")]
chemin_parquet = hf_hub_download(REPO, fichiers[0], repo_type="dataset")
brut = pd.read_parquet(chemin_parquet, engine="fastparquet")

# Échantillon aléatoire mais reproductible
brut = brut.sample(n=N, random_state=42)

# On range les images dans un dossier
dossier = Path("data/images")
dossier.mkdir(parents=True, exist_ok=True)

lignes = []
for _, item in brut.iterrows():
    # Selon la lecture, l'image est rangée de deux façons possibles
    octets = item["image.bytes"] if "image.bytes" in brut.columns else item["image"]["bytes"]
    chemin = dossier / f"{item['id']}.jpg"
    Image.open(io.BytesIO(octets)).convert("RGB").save(chemin)
    lignes.append({
        "id": item["id"],
        "nom": item["productDisplayName"],
        "categorie": item["articleType"],
        "couleur": item["baseColour"],
        "image": str(chemin),
    })

# On enregistre les infos produits dans un tableau (CSV)
df = pd.DataFrame(lignes)
df.to_csv("data/produits.csv", index=False)
print(df.head())
print(len(df), "produits enregistrés")