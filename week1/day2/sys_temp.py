import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("api key not found")

client = Groq(api_key = my_api_key)

model = "openai/gpt-oss-120b"
role = "user"
prompt = "suggest a name for food company"

message_system ={
    "role" : "system",
    "content" : "you are brand manager who suggest name for a company, suggest name in one word only"
}

message ={
    "role" : role,
    "content"  : prompt
}

messages = [message_system, message]

response = client.chat.completions.create(model = model, messages = messages, temperature = 2)

# print(response)

print("################################################################")

answer = response.choices[0].message.content
print(answer)