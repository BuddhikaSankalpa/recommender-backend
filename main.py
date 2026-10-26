from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

# FastAPI app එක ආරම්භ කිරීම
app = FastAPI()

# Frontend එකට (React) මෙම API එකට කතා කිරීමට අවසර දීම (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # සැබෑ වෙබ් අඩවියකදී මෙය ඔබගේ Frontend URL එක විය යුතුය
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading Models and Data... Please wait.")

# 1. Models සහ Data Load කරගැනීම (ඔබේ ෆෝල්ඩරයේ නමට අදාළව Paths සකසා ඇත)
MODEL_DIR = "models/"

movies_df = pd.read_csv(f"{MODEL_DIR}movies_data.csv")
movie_matrix = sparse.load_npz(f"{MODEL_DIR}movie_matrix.npz")

tv_df = pd.read_csv(f"{MODEL_DIR}tv_data.csv")
tv_matrix = sparse.load_npz(f"{MODEL_DIR}tv_matrix.npz")

print("Loading Complete! Server is ready.")

# 2. Frontend එකෙන් එන Request එකේ තිබිය යුතු දත්ත ආකෘතිය
class SearchQuery(BaseModel):
    title: str
    media_type: str # "movie" හෝ "tv" විය යුතුය
    top_k: int = 5

# 3. ප්‍රධාන API Endpoint එක (Recommendation Logic)
@app.get("/")
async def root():
    return {"message": "Welcome to the Movie/TV Recommender API!"}

# 3. Principal API Endpoint
@app.post("/recommend")
async def get_recommendations(query: SearchQuery):
    title = query.title.strip().casefold()
    
    if query.media_type == "movie":
        data = movies_df
        matrix = movie_matrix
    elif query.media_type == "tv":
        data = tv_df
        matrix = tv_matrix
    else:
        raise HTTPException(status_code=400, detail="media_type must be 'movie' or 'tv'")

    # නම (Title) දත්ත ගොනුවේ තිබේදැයි සෙවීම
    matches = data[data["title"].astype(str).str.casefold() == title].index

    if len(matches) == 0:
        raise HTTPException(status_code=404, detail=f"'{query.title}' not found in {query.media_type}s.")

    source_idx = matches[0]

    # Cosine Similarity ගණනය කිරීම
    scores = cosine_similarity(matrix[source_idx], matrix).ravel()
    scores[source_idx] = -1 # සෙවූ එකම නැවත පැමිණීම වැළැක්වීම

    top_indices = scores.argsort()[::-1][:query.top_k]

    # ප්‍රතිඵල JSON (Dictionary List) එකක් ලෙස සැකසීම
    results = []
    for idx in top_indices:
        movie_data = data.iloc[idx]
        results.append({
            "id": int(movie_data["id"]),
            "title": str(movie_data["title"]),
            "genres": str(movie_data["genres"]),
            "overview": str(movie_data["overview"]),
            "poster_path": str(movie_data["poster_path"]) if pd.notna(movie_data["poster_path"]) else None,
            "similarity": float(scores[idx])
        })

    return {"results": results}