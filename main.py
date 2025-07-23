from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import requests
import os
import numpy as np
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Environment
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")  # Hugging Face token
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

def simulate_embedding(text):
    response = requests.post(
        "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2",
        headers={"Authorization": f"Bearer {HF_API_KEY}"},
        json={"inputs": text}
    )
    response.raise_for_status()
    embedding = response.json()
    return embedding[0]  # HuggingFace returns [ [vector] ]

@app.post("/chat")
async def chat(request: Request):
    try:
        body = await request.json()
        user_question = body.get("question")

        if not user_question:
            return {"error": "No question provided."}

        print("User question:", user_question)

        # Get embedding
        question_embedding = simulate_embedding(user_question)
        print("Simulated embedding vector (first 5 dims):", question_embedding[:5])

        # Fetch chunks from Supabase
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json"
        }

        supabase_url = f"{SUPABASE_URL}/rest/v1/{TABLE_NAME}?select=text,embedding"
        print("Calling Supabase URL:", supabase_url)

        res = requests.get(supabase_url, headers=headers)
        print("Supabase status:", res.status_code)

        if res.status_code != 200:
            return {"error": f"Supabase request failed", "status": res.status_code, "details": res.text}

        all_rows = res.json()

        if not all_rows:
            return {"error": "No chunks found in Supabase."}

        relevant_chunks = []
        for row in all_rows:
            embedding = row.get("embedding")
            if not embedding or len(embedding) != EMBEDDING_DIM:
                continue
            score = float(np.dot(question_embedding, embedding) /
                          (np.linalg.norm(question_embedding) * np.linalg.norm(embedding)))
            relevant_chunks.append({
                "text": row["text"],
                "score": score
            })

        if not relevant_chunks:
            return {"error": "No valid embeddings found for scoring."}

        sorted_chunks = sorted(relevant_chunks, key=lambda x: x["score"], reverse=True)[:3]
        context = "\n\n".join(chunk["text"] for chunk in sorted_chunks)

        # Prompt
        prompt = f"""You are a helpful skincare assistant. Answer the user's question based on the context below.

Context:
{context}

Question:
{user_question}

Answer:"""

        print("Calling Groq with prompt:")
        print(prompt[:300] + "..." if len(prompt) > 300 else prompt)

        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="mixtral-8x7b-32768",
            temperature=0.7
        )

        answer = response.choices[0].message.content
        print("LLM Answer:", answer[:300] + "..." if len(answer) > 300 else answer)

        return {"answer": answer}

    except Exception as e:
        print("ERROR OCCURRED:", str(e))
        return {"error": "Internal Server Error", "details": str(e)}

@app.get("/")
def read_root():
    return {"message": "Skincare RAG API is live 🚀"}
