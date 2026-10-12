import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
text = "virat kohli is goat"

# === it converts the text into vector ===
# embedding = model.encode(text)
# print(embedding[:10])


# === it gives how similar two vectors are based on their direction ===
def cosine_similarity(a,b):
    return np.dot(a,b)/ (np.linalg.norm(a) * np.linalg.norm(b))


v1 = model.encode("i love programming")
v2 = model.encode("i enjoy coading")
v3 = model.encode("dog is a animal")

print(cosine_similarity(v1,v2))  # 0.43 -> as it is more similar the answer is more closer to 1
print(cosine_similarity(v1,v3))  # 0.12 -> as it is different it is smaller
