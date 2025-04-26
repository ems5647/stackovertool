from dotenv import load_dotenv
import os
import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from pymilvus import Collection, connections
import openai
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()  # 👈 This MUST exist!

# your endpoint routes below...
@app.get("/health")
def health_check():
    return {"status": "ok"}

# more code...