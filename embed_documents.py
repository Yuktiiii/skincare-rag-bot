print("🐍 Script is running!")

from sentence_transformers import SentenceTransformer
import os
import re
import json

def embed_documents():
    print("🔄 Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Model loaded.")

    print("📂 Reading skincare_faqs.md...")
    try:
        with open("data/skincare_faqs.md", "r", encoding="utf-8") as f:
            content = f.read()
        print("✅ File read successfully.")
    except FileNotFoundError:
        print("❌ skincare_faqs.md not found.")
        return

    chunks = re.split(r"### ", content)
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    print(f"🔹 Found {len(chunks)} content chunks.")

    data = []
    for chunk in chunks:
        print(f"🔍 Embedding chunk: {chunk[:40]}...")
        embedding = model.encode(chunk).tolist()
        data.append({
            "text": chunk,
            "embedding": embedding
        })

    with open("data/embedded_chunks.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("✅ Embeddings saved to /data/embedded_chunks.json")

if __name__ == "__main__":
    embed_documents()




