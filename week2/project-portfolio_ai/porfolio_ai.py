import os
from pathlib import Path
from groq import Groq
from time import sleep
from dotenv import load_dotenv
from pypdf import PdfReader
from pydantic import BaseModel
import json


load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("api key not found")

client = Groq(api_key= my_api_key)
model = "openai/gpt-oss-120b"




#llm call function we can use llm with different prompt, you can call this function whenever the llm call required
def ask_llm(user_prompt, system_prompt):

    user_message ={
        "role" : "user",
        "content" : user_prompt
    }

    system_message={
        "role" : "system",
        "content" : system_prompt
    }
    

    messages =[user_message, system_message]
    response = client.chat.completions.create(model=model, messages= messages)
    answer=response.choices[0].message.content
    return answer


# to convert the pdf information into string 
def read_pdf(file_path):
    reader = PdfReader(file_path)

    text = ""

    for page in reader.pages:
        text += page.extract_text() + "\n"

    return text 



# step 1 : to extract the information from student profile
student_profile = read_pdf("profiles/Arjun_Sharma_Student_Profile.pdf")


def extract_profile(student_profile):
    print("step 1")
    system_prompt = """
    you are a very intelligent ai made to extract the information from the given profile, so extract all the 
    essential information that can be asked by an hr or required for hr dont add imaginary information only give
    the information that are present in the profile with honesty dont lie during giving the information 
    output format = dont give the information full of paragraphs give a systematic answer so that anyone 
    can see essential information
    """

    user_prompt = f"extract the information from this profile {student_profile}"

    return ask_llm(user_prompt, system_prompt)







#step2 : to extract the information from the resume in json format / to make the resume parser
resume_text = read_pdf("resumes/senior_software_engineer_resume_sample (1).pdf")


#json object for resume and experience
class experience(BaseModel):
    company : str | None= None
    role: str | None = None
    duration: str | None = None
    description: str | None = None
    skills_used: list[str] = []


class Resume(BaseModel):
    name : str | None = None
    education : list[str] = []
    experiences : list[experience] = []
    total_year_experience : float | None=None
    skills : list[str] = []
    projects : list[str] = []
    certifications : list[str] = []

resume_schema = Resume.model_json_schema()
    

#to extract the information of resume in the json format
def extract_resume(resume_text):
    print("step 2")
    system_prompt = f"""
        You are an expert resume parser.
    
        Extract information from the resume based on its meaning,
        not only based on exact section headings.
    
        Different resumes may use different headings.
    
        For example:
        - Experience
        - Professional Experience
        - Work History
        - Employment
        - Internships
    
        These may all contain relevant experience.
    
        Skills may also appear in the skills section, work experience,
        internships or projects.
    
        Return ONLY valid JSON matching this schema:
    
        {resume_schema}
    
        Important rules:
    
        1. Do not invent information.
        2. If a value is not available, return null.
        3. If a list has no information, return an empty list.
        4. Include internships inside experiences.
        5. Extract skills mentioned across the entire resume.
        """

    user_prompt = f"parse the information from this text {resume_text} "

    return ask_llm(user_prompt, system_prompt)



profile_info =extract_profile(student_profile)
# print(profile_info)
sleep(1)
resume_info = extract_resume(resume_text)
# print(resume_info)
sleep(1)


#step3 : question and answer system and conversation history


     
def ask_question(profile_info, resume_info, Que ):
    
    system_prompt = f"""
    you are a expert ai assistant that can give the answer asked by the hr from the profile of the candidate
    and the resume of the candidate. if hr ask any question please give the answer from the information given 
    dont generate new information and strictlt dont create any answer by your own give honest answer from provided
    information. give the answer to this question {Que}
    ex = HR : how many percentage canditate get in class 10th?
         ans : 90.80%
    give honest and clear answer strictly from the information given to you.
    output format : dont give answer in answer : answer just give the answer 
    if there is any irrelevant question which is not professional and not related to the questions usually asked
    by hr return "irrelevant ques"
    
    """
    user_prompt = f"take the information to give the answer from {profile_info} and {resume_info} "

    return ask_llm(user_prompt, system_prompt)


var = True
while(var):
    Que = input("enter the ques : ")
    if(Que=="stop"):
        break
    ans = ask_question(profile_info, resume_info, Que)
    print(ans)

