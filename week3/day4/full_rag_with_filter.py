import os
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, MatchAny, PayloadSchemaType
from groq import Groq
import json

load_dotenv()

Qdrant_api = os.getenv("QDRANT_API_KEY")
Qdrant_URL = os.getenv("QDRANT_URL")

client = QdrantClient(url = Qdrant_URL, api_key = Qdrant_api)


# create collection in qdrant 

Collection_name = "data"
embedding_size = 384

if client.collection_exists(Collection_name):
    client.delete_collection(Collection_name)
    print(f"{Collection_name} collection is deleted")


client.create_collection(
    collection_name = Collection_name,
    vectors_config = VectorParams(size=embedding_size, distance= Distance.COSINE)
)

client.create_payload_index(
    collection_name=Collection_name,
    field_name="category",
    field_schema=PayloadSchemaType.KEYWORD,
)


print(f"{Collection_name} is created")


# load the data

with open("data.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

# create embedding 

model = SentenceTransformer("all-MiniLM-L6-v2")
texts = [document["text"] for document in documents]

embeddings = model.encode(texts)

# create qdrant points 

points_info = []

for i, embedding in enumerate(embeddings):
    Point = PointStruct(
        id = i+1,
        vector = embedding.tolist(),
        payload = {"text" : documents[i]}
    )

    points_info.append(Point)

# upload points in collection to qdrant

client.upsert(collection_name= Collection_name, points = points_info)


# search qdrant

# def search(query,top_k=3):

#     query_vector = model.encode(query).tolist()

#     results = client.query_points(
#         collection_name = Collection_name,
#         query = query_vector,
#         limit = top_k,
#         with_payload = True
#     ).points

#     return results


def search_with_filter(query,query_filter,top_k=3):

    query_vector = model.encode(query).tolist()

    results = client.query_points(
        collection_name = Collection_name,
        query = query_vector,
        limit = top_k,
        with_payload = True,
        query_filter = query_filter
    ).points

    return results

leave_query_filter = Filter(
    must = [
        FieldCondition(
            key = "category",
            match = MatchValue(value = "leave")
        )
    ]
)

my_api_key = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key = my_api_key)

# def ask llm

def ask_llm(question,context):

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



question = "How many paid leaves do i get?" 

final_results = search_with_filter(question,leave_query_filter,top_k=3)

context = "\n".join(
    result.payload["text"]
    for result in final_results
)


answer = ask_llm(question,context)

print(answer)
