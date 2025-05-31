# 🎬 Movie Q&A Agent — Hybrid RAG + Live Search Powered by OpenRouter & ChromaDB

A smart, Retrieval-Augmented Generation (RAG) based chatbot that answers natural language questions about movies and TV series using real movie metadata **and** live internet search fallback. Built with ChromaDB for semantic vector search over the [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata), OpenRouter for LLM access, and ARES API for live internet queries.

---

## 🚀 Features

- 🔍 **Hybrid RAG + Live Search**: Answers come from vector similarity search over TMDB data **or** live internet results when data isn’t found locally.
- 🧠 **Query Classification**: Routes queries intelligently to:
  - Movie/TV metadata search (ChromaDB)
  - OpenRouter LLM chat
  - ARES live internet search
- 🌐 **Flexible LLM Access**: Use GPT-4, Mixtral, Claude, or other OpenRouter-supported models seamlessly.
- 🎥 **Real Movie Dataset**: Integrates TMDB metadata with cast, crew, genres, and overviews.
- 🛠️ **Easy to Extend**: Modular code for classification, retrieval, generation, and web search fallback.

---

## 📂 Dataset

This project uses the following Kaggle dataset files:

- `tmdb_5000_movies.csv`: Core movie metadata including titles, genres, and plot summaries.
- `tmdb_5000_credits.csv`: Cast and crew details.

> 📥 [Download the TMDB Movie Dataset from Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

---

## ⚙️ How It Works

1. **Data Preparation**  
   - Merge movies and credits datasets  
   - Clean and extract key fields (title, genres, overview, top cast & crew)  
   - Build single-document text blobs for each movie/series  

2. **Embedding & Vector DB**  
   - Use `sentence-transformers` (`all-MiniLM-L6-v2`) to create vector embeddings for each movie document  
   - Store embeddings and metadata in ChromaDB for fast semantic search  

3. **Query Classification**  
   - Use an LLM-based router prompt to classify queries into:  
     - `MOVIE_QUERY` (search in ChromaDB)  
     - `INTERNET_QUERY` (fetch live info using ARES API)  
     - `GENERIC_CHAT` (fallback to chat with LLM)  

4. **Query Routing & Response**  
   - For movie queries, retrieve top relevant movie chunks and generate answer via OpenRouter  
   - For internet queries, forward the question to ARES live search API and return clean, readable responses  
   - For generic queries, respond with LLM chat completions  

---

## 🧪 Example Usage

```python
query = "When is the next season of Stranger Things coming?"

response = movie_agent(query)
print(response['recommendations'])

# Clean output:
# The next season of "Stranger Things," Season 5, is expected in 2025.
