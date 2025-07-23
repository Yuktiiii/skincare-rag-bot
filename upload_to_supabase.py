import pandas as pd
import requests
import ast

# ⬇️ Replace with your actual values
SUPABASE_URL = "https://binfdetyhjqcoubzkkvc.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJpbmZkZXR5aGpxY291Ynpra3ZjIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTMxODcyMDcsImV4cCI6MjA2ODc2MzIwN30.SoDYmOeLD0GRcKip-oui5W4M5N7bI1ALnxQOguuqr3M"
TABLE_NAME = "chunks"

# Load CSV
df = pd.read_csv("data/embedded_chunks.csv")
print(f"📄 Loaded {len(df)} rows from CSV.")

# Set headers
headers = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json"
}

# Upload row by row
for i, row in df.iterrows():
    try:
        # Convert embedding string to list
        embedding = ast.literal_eval(row['embedding'])

        payload = {
            "text": row['text'],
            "embedding": embedding
        }

        response = requests.post(
            f"{SUPABASE_URL}/rest/v1/{TABLE_NAME}",
            headers=headers,
            json=payload
        )

        if response.status_code == 201:
            print(f"✅ Row {i+1} uploaded.")
        else:
            print(f"❌ Row {i+1} failed: {response.status_code} - {response.text}")

    except Exception as e:
        print(f"🔥 Error at row {i+1}: {str(e)}")


