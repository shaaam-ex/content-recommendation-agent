# Cell 1: Import necessary libraries and initialize global variables

import pandas as pd
import ast
import re
import json
import requests

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# For calling the LLM router (OpenRouter client or OpenAI client)
from openai import OpenAIError

# Assuming open_router_client and ares_api_key are configured elsewhere in your environment

print("Libraries imported and globals ready.")

# Time to give this output to an LLM
import openai

open_router_client = openai.OpenAI(
    api_key="api-key",
    base_url="https://openrouter.ai/api/v1" 
)

# Cell 2: Load datasets, merge and clean relevant fields

movies_df = pd.read_csv('../data/tmdb_5000_movies.csv')
credits_df = pd.read_csv('../data/tmdb_5000_credits.csv')

# Merge on movie title
df = movies_df.merge(credits_df, on='title')

# Helper function to parse JSON-like strings and extract 'name' fields
def extract_names(json_str, top_n=None):
    try:
        items = ast.literal_eval(json_str)
        names = [item['name'] for item in items]
        if top_n:
            names = names[:top_n]
        return ', '.join(names)
    except (ValueError, SyntaxError):
        return ''

# Apply the helper to genres, cast, and crew columns
df['genres_clean'] = df['genres'].apply(lambda x: extract_names(x))
df['cast_clean'] = df['cast'].apply(lambda x: extract_names(x, top_n=5))
df['crew_clean'] = df['crew'].apply(lambda x: extract_names(x, top_n=5))

print("Data loaded and cleaned.")

# Cell 3: Build a single descriptive text block per movie for embedding

def build_movie_document(row):
    return f"""
    Title: {row['title']}
    Genres: {row['genres_clean']}
    Overview: {row['overview']}
    Cast: {row['cast_clean']}
    Crew: {row['crew_clean']}
    """

df['document'] = df.apply(build_movie_document, axis=1)

print("Movie documents prepared.")

# Cell 4: Generate sentence embeddings and store them in ChromaDB

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create list of documents
documents = df['document'].tolist()

# Generate embeddings for all documents (can take a moment)
embeddings = model.encode(documents, show_progress_bar=True)

# Initialize Chroma client and create collection
client = chromadb.Client(Settings())
collection = client.create_collection(name="movies")

# Add documents and embeddings to ChromaDB collection
collection.add(
    documents=documents,
    embeddings=embeddings,
    ids=[str(i) for i in range(len(documents))]
)

print("Embeddings generated and stored in ChromaDB.")

# Cell 5: Define the ARES fallback function to get internet content when needed

def get_internet_content(user_query: str):
    print("Fetching response from ARES live search...")

    url = "https://api-ares.traversaal.ai/live/predict"
    payload = {"query": [user_query]}
    headers = {
        "x-api-key": 'ares',
        "content-type": "application/json"
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json().get('data', {}).get('response_text', "No response received.")

    except requests.exceptions.HTTPError as http_err:
        return f"HTTP error occurred: {http_err}"

    except requests.exceptions.RequestException as req_err:
        return f"Request error occurred: {req_err}"

    except Exception as err:
        return f"An unexpected error occurred: {err}"

# Cell 6: Define the query classifier using GPT-4o

# Cell 6: Define the query classifier using GPT-4o
def classify_query(user_query: str):
    router_system_prompt = f"""
    You are a professional query router. Categorize the user's input:
    1. "MOVIE_QUERY": For movie/series recommendations.
    2. "INTERNET_QUERY": For fresh internet data (reviews, trailers).
    3. "GENERIC_CHAT": For casual or unrelated chat.

    Examples of GENERIC_CHAT:
    - "How are you?"
    - "What's up?"
    - "Tell me a joke."

    Examples of NON-GENERIC queries (should return "No"):
    - "Who directed Interstellar?"
    - "What is Stranger Things about?"
    - "When is the next season of Breaking Bad?"

    Respond ONLY with a valid JSON in this format:
    {{
        "action": "MOVIE_QUERY" or "INTERNET_QUERY" or "GENERIC_CHAT",
        "reason": "brief reason"
    }}

    Query: "{user_query}"
    """

    try:
        response = open_router_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": router_system_prompt}],
            max_tokens=50,  # limit tokens to reduce cost
            temperature=0
        )
        json_text = re.search(r"\{.*\}", response.choices[0].message.content, re.DOTALL).group()
        classification = json.loads(json_text)
        return classification

    except Exception as e:
        print(f"Error during classification: {e}")
        return {"action": "GENERIC_CHAT", "reason": "Fallback due to error."}


# Cell 7: Search ChromaDB with embedding similarity

def search_chromadb(user_query: str, collection, top_k=5):
    query_embedding = model.encode([user_query])[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    # results['documents'][0] contains the matched documents
    return results['documents'][0] if results['documents'] else []

# Cell 8: The main agent function combining classification, search, and fallback

def movie_agent(user_query: str):
    print(f"Received user query: {user_query}")

    # Step 1: Classify query
    classification = classify_query(user_query)
    action = classification.get('action')
    reason = classification.get('reason')
    print(f"Classification result: {action} (Reason: {reason})")

    if action == "MOVIE_QUERY":
        results = search_chromadb(user_query, collection)
        if results:
            print("Found recommendations in ChromaDB.")
            return {
                "source": "ChromaDB",
                "recommendations": results
            }
        else:
            print("No good matches found locally. Falling back to ARES.")
            fallback = get_internet_content(user_query)
            return {
                "source": "ARES",
                "recommendations": fallback
            }

    elif action == "INTERNET_QUERY":
        print("Routing query to ARES live internet search.")
        fallback = get_internet_content(user_query)
        return {
            "source": "ARES",
            "recommendations": fallback
        }

    else:
        print("Generic chat detected, no movie recommendations.")
        return {
            "source": "Agent",
            "recommendations": "Sorry, I can only help with movie/series queries!"
        }

def format_response(response, classification_action):
    if classification_action == "MOVIE_QUERY":
        # response['recommendations'] is a list of structured movie docs
        return clean_recommendations(response['recommendations'])
    elif classification_action == "INTERNET_QUERY":
        # response['recommendations'] is a plain text string from ARES
        return clean_ares_response(response['recommendations'])
    else:
        return "No relevant data found."

def clean_recommendations(raw_recs):
    cleaned_recs = []
    for rec in raw_recs:
        lines = rec.strip().split('\n')
        title = genres = overview = ''
        for line in lines:
            line = line.strip()
            if line.startswith('Title:'):
                title = line[len('Title:'):].strip()
            elif line.startswith('Genres:'):
                genres = line[len('Genres:'):].strip()
            elif line.startswith('Overview:'):
                overview = line[len('Overview:'):].strip()
        if len(overview) > 200:
            overview = overview[:197] + '...'
        cleaned = f"Title: {title}\nGenres: {genres}\nOverview: {overview}\n"
        cleaned_recs.append(cleaned)
    return "\n---\n".join(cleaned_recs)

def clean_ares_response(text):
    # Remove multiple newlines and trim
    import re
    text = re.sub(r'\n\s*\n', '\n\n', text.strip())
    return text



from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/ask")
async def ask_movie_bot(req: QueryRequest):
    query = req.query
    try:
        response = movie_agent(query)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8100)

# uvicorn api:app --host 0.0.0.0 --port 8100