# On importe les outils dont on a besoin
from sentence_transformers import SentenceTransformer, util
from PIL import Image

# On charge le modèle CLIP (téléchargé automatiquement la première fois)
model = SentenceTransformer("clip-ViT-B-32")

# On ouvre nos 3 images
noms = ["chaise1.jpg", "chaise2.jpg", "tasse.jpg"]
images = [Image.open(f"images/{nom}") for nom in noms]

# CLIP transforme chaque image en "empreinte" : une liste de nombres
embeddings = model.encode(images, convert_to_tensor=True)
print("Une empreinte contient", embeddings.shape[1], "nombres")

# On compare chaque paire d'images : plus le score est proche de 1, plus elles se ressemblent
scores = util.cos_sim(embeddings, embeddings)
for i in range(len(noms)):
    for j in range(i + 1, len(noms)):
        print(f"{noms[i]} vs {noms[j]} : {scores[i][j]:.2f}")

        # CLIP comprend aussi le texte : on compare chaque image à des descriptions
textes = ["a photo of a chair", "a photo of a mug"]
emb_textes = model.encode(textes, convert_to_tensor=True)
sim_texte = util.cos_sim(embeddings, emb_textes)

print("\nCe que CLIP reconnaît :")
for i, nom in enumerate(noms):
    meilleur = sim_texte[i].argmax()
    print(f"{nom} → {textes[meilleur]}")