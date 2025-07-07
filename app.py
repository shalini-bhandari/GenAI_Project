from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import requests

# Load .env
load_dotenv()
HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Smaller free models
SUMMARIZATION_MODEL = "google/flan-t5-small"
REPLY_MODEL = "google/flan-t5-small"

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Complaint(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "Backend with Hugging Face is working with smaller models!"}

@app.post("/summarize")
def summarize(complaint: Complaint):
    url = f"https://api-inference.huggingface.co/models/{SUMMARIZATION_MODEL}"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    prompt = f"Summarize this: {complaint.text}"
    payload = {"inputs": prompt}
    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        output = response.json()
        summary = output[0]['generated_text']
        return {"summary": summary}
    else:
        return {"error": response.json()}

@app.post("/generate-response")
def generate_response(complaint: Complaint):
    url = f"https://api-inference.huggingface.co/models/{REPLY_MODEL}"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    prompt = f"Write a short, polite, professional reply to this customer complaint: {complaint.text}"
    payload = {"inputs": prompt}
    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        output = response.json()
        reply = output[0]['generated_text']
        return {"response": reply}
    else:
        return {"error": response.json()}
