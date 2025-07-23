from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import requests
import os
import ast
import numpy as np
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Environment
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TABLE_NAME = "chunks"
EMBEDDING_DIM = 384

# Groq LLM client
groq_client = Groq(api_key=GROQ_API_KEY)

# FastAPI app
app = FastAPI()

# CORS (for frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    user_question = body.get("question")

    if not user_question:
        return {"error": "No question provided."}

    # Call embedding API (optional; assume already embedded for now)
    # Here we simulate embedding locally — you can replace with API call
    question_embedding = simulate_embedding(user_question)

    # Fetch chunks
    res = requests.get(
        f"{SUPABASE_URL}/rest/v1/{TABLE_NAME}?select=text,embedding",
        headers={
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}"
        }
    )
    all_rows = res.json()

    # Parse and score
    for row in all_rows:
        row["embedding"] = ast.literal_eval(row["embedding"])
        row["score"] = float(np.dot(question_embedding, row["embedding"]) /
                             (np.linalg.norm(question_embedding) * np.linalg.norm(row["embedding"])))
    sorted_chunks = sorted(all_rows, key=lambda x: x["score"], reverse=True)[:3]
    context = "\n\n".join(chunk["text"] for chunk in sorted_chunks)

    # Prompt
    prompt = f"""You are a helpful skincare assistant. Answer the user's question based on the context below.

Context:
{context}

Question:
{user_question}

Answer:"""

    response = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="mixtral-8x7b-32768",
        temperature=0.7
    )

    return {"answer": response.choices[0].message.content}

def simulate_embedding(text):
    # Fake placeholder vector just to keep structure running — REPLACE with real vector if needed
    return np.ones(EMBEDDING_DIM).tolist()
