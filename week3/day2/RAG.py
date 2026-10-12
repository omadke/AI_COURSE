import numpy as np
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
from pathlib import Path
from groq import Groq

load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("api key not found")

client = Groq(api_key= my_api_key)
groq_model = "openai/gpt-oss-120b"


def ask_llm(query,context):

    sys_prompt = f"give the answer in one line. do not give imaginary answer give the answer strictly from this context : {context}"

    sys_prompt ={
        "role" : "system",
        "content" : sys_prompt
    }


    user_msg ={
        "role" : "user",
        "content" : query
    }

    messages = [user_msg]
    response = client.chat.completions.create(model= groq_model, messages= messages)
    answer = response.choices[0].message.content
    return answer


# ====== HOW RAG SYSTEM WORKS =======

documents = [
"we have paid leave of 24 days."
"we have lots of work in office."
"we work 7 hours a day."
"out company is software based company."
]

model = SentenceTransformer("all-MiniLM-L6-v2")
document_embedding = model.encode(documents)

def cosine_similarity(a,b):
    return np.dot(a,b)/ (np.linalg.norm(a) * np.linalg.norm(b))


def retrives(query_embedding):
    scores =[]
    for i,document in enumerate(document_embedding):
        score = cosine_similarity(query_embedding, document)
        scores.append((score, documents[i]))

    scores.sort(reverse= True)
    return scores[0]


query = "how many holidays we get"
query_embedding = model.encode(query)
score,context = retrives(query_embedding)

ans = ask_llm(query,context)
print(ans)




