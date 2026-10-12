import os 

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from groq import Groq

# it loads variables from .env
load_dotenv()

Quadrant_URL = os.getenv("QDRANT_URL")
Quadrant_api_key = os.getenv("QDRANT_API_KEY")
Groq_api_key = os.getenv("GROQ_API_KEY")



# ===========================================================
# part2 - connect to qdrant
# ===========================================================

client = QdrantClient(url = Quadrant_URL, api_key = Quadrant_api_key)

print("connected to qdrant cloud!")


# ===============================================================
#  part3 - create qdrant collection 
# ===============================================================

Collection_name = "knowledge"
Embedding_size = 384

# delete collection if it already exist
if client.collection_exists(Collection_name):
    client.delete_collection(Collection_name)
    print(f"deleted existing collection : {Collection_name}")


# create collection
client.create_collection(
    collection_name = Collection_name,
    vectors_config = VectorParams(size= Embedding_size, distance = Distance.COSINE),
    )

print(f"created collection : {Collection_name}")
print(f"vector size : {Embedding_size}")
print("Distance : COSINE")


# =========================================================================
# part4 - load our knowledge 
# =========================================================================

with open("knowledge.txt", "r", encoding="utf-8") as f:
    documents = [
        line.strip()
        for line in f
        if line.strip()
    ]
# ["line 1 ", "line2", "line3"....]
print(f"Loaded {len(documents)} documents")


# ============================================================================
# part5 - create embedding 
# ============================================================================

print("loading embedding model....")

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(documents)


# ========================================================================
# part6 - Create Qdrant Points 
# ========================================================================

points = []

for i,embedding in enumerate(embeddings):
    Point = PointStruct( 
        id = i+1,
        vector = embedding.tolist(),

        payload = {"text" : documents[i]}
    )

    points.append(Point)


# ============================================================================
# part7 - upload to Qdrant
# ============================================================================

client.upsert(collection_name = Collection_name, points = points)

print("uploaded the document to the qdrant")

# =============================================================================
# part8 - search qdrant
# =============================================================================

def search(query, top_k=3):

    query_vector = model.encode(query).tolist()

    # search qdrant for similar vector 
    results = client.query_points(
        collection_name = Collection_name,
        query = query_vector,
        limit = top_k,
        with_payload = True
    ).points

    return results


# ==============================================================================
# part9 - test search
# ==============================================================================

query = "how many vacation days do i get ?"

results = search(query, top_k=3)

for result in results:
    print(result.payload["text"])
    print()


# ============================================================
# PART 10 — CONNECT TO GROQ
# ============================================================

my_api_key = os.getenv("GROQ_API_KEY")
groq_client = Groq(
    api_key= my_api_key
)


# ============================================================
# PART 11 — ASK THE LLM
# ============================================================

def ask_llm(question, context):

    prompt = f"""
Answer the question using only the information provided below.

Context:
{context}

Question:
{question}

If the answer is not present in the context, say:
"I don't know based on the provided information."
"""

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content


# ============================================================
# PART 12 — COMPLETE RAG PIPELINE
# ============================================================

question = "How many vacation days do I get?"

final_results = search(question, top_k=3)


# Extract text from the search results
context = "\n".join(
    result.payload["text"]
    for result in final_results
)
    
    



answer = ask_llm(question, context)


print("\nFinal Answer:")
print(answer)