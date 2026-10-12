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



from pydantic import BaseModel
class Ticket(BaseModel) :
    name : str
    email : str
    issue : str

schema=Ticket.model_json_schema()

response_format={
    "type": "json_object"
}

system_prompt = f"""
Extract the personal information from the ticket strictly based on this schema and give a json output.
{schema}
"""



text = "hi my name is om. and my address is nashik. i have an iphone which is not working. and my email is abc@gamil.com. and my phone number is 7387 "
prompt = f"""
this is a customer ticket please extract the information from this {text}
"""

sys_message ={
    "role" : "system",
    "content" : system_prompt
}


message ={
    "role" : role,
    "content"  : prompt
}

messages = [sys_message, message]

response = client.chat.completions.create(model = model, messages = messages, response_format= response_format)



print("################################################################")

answer = response.choices[0].message.content
print(answer)

# how to read it 
import json
raw_json = answer
data_file = json.loads(raw_json)
ticket = Ticket(**data_file)

print(ticket.name)
print(ticket.email)
print(ticket.issue)