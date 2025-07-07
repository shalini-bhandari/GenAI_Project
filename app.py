from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import requests

load_dotenv()
HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

SUMMARIZATION_MODEL = "facebook/bart-large-cnn"
REPLY_MODEL = "tiiuae/falcon-7b-instruct"


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

    try:
        response = requests.post(url, headers=headers, json=payload)

        # Check content type before parsing
        if "application/json" not in response.headers.get("Content-Type", ""):
            raise HTTPException(status_code=502, detail="Non-JSON response from Hugging Face")

        output = response.json()

        # Hugging Face summarizers return summary_text, not generated_text
        if isinstance(output, list) and 'summary_text' in output[0]:
            summary = output[0]['summary_text']
            return {"summary": summary}
        else:
            raise HTTPException(status_code=500, detail="Unexpected summarization output format")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during summarization: {str(e)}")


@app.post("/generate-response")
def generate_response(complaint: Complaint):
    url = f"https://api-inference.huggingface.co/models/{REPLY_MODEL}"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    prompt = f"Write a short, polite, professional reply to this customer complaint: {complaint.text}"
    payload = {"inputs": prompt}

    try:
        response = requests.post(url, headers=headers, json=payload)

        print("Status code:", response.status_code)
        print("Content-Type:", response.headers.get("Content-Type", ""))
        print("Raw text:", response.text)

        if "application/json" not in response.headers.get("Content-Type", ""):
            raise HTTPException(status_code=502, detail=f"Non-JSON response from Hugging Face: {response.text}")

        output = response.json()

        if isinstance(output, list) and 'generated_text' in output[0]:
            return {"response": output[0]['generated_text']}
        else:
            raise HTTPException(status_code=500, detail="Unexpected response format.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during response generation: {e}")
