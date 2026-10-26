import os
import re
import asyncio
import json
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import urlopen
from typing import Literal

from fastapi import FastAPI, HTTPException, Path, Query
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


def shelf_items(data, media_type):
    # Keep the first row for each name, matching /recommend's title lookup.
    seen = set()
    items = []
    for row in data.itertuples(index=False):
        title = str(row.title).strip()
        key = title.casefold()
        if not title or key in seen:
            continue
        seen.add(key)
        items.append({
            "id": int(row.id),
            "title": title,
            "media_type": media_type,
            "genres": str(row.genres) if pd.notna(row.genres) else "",
            "poster_path": str(row.poster_path) if pd.notna(row.poster_path) else None,
        })
    return items


SHELF_MOVIES = shelf_items(movies_df, "movie")
SHELF_TV = shelf_items(tv_df, "tv")
SHELF_ALL = []
seen_shelf_titles = set()
for index in range(max(len(SHELF_MOVIES), len(SHELF_TV))):
    for catalogue in (SHELF_MOVIES, SHELF_TV):
        if index < len(catalogue):
            item = catalogue[index]
            key = item["title"].casefold()
            if key not in seen_shelf_titles:
                seen_shelf_titles.add(key)
                SHELF_ALL.append(item)

print("Loading Complete! Server is ready.")

# 2. Frontend එකෙන් එන Request එකේ තිබිය යුතු දත්ත ආකෘතිය
class SearchQuery(BaseModel):
    title: str
    media_type: str # "movie" හෝ "tv" විය යුතුය
    top_k: int = 5
    tmdb_id: int | None = None

# 3. ප්‍රධාන API Endpoint එක (Recommendation Logic)
@app.get("/")
async def root():
    return {"message": "Welcome to the Movie/TV Recommender API!"}


@app.get("/shelf")
async def get_shelf(media_type: Literal["all", "movie", "tv"] = "all", offset: int = Query(default=0, ge=0)):
    catalogue = {"all": SHELF_ALL, "movie": SHELF_MOVIES, "tv": SHELF_TV}[media_type]
    items = catalogue[offset:offset + 5]
    return {"items": items, "has_more": offset + len(items) < len(catalogue)}


OMDB_URL = "https://www.omdbapi.com/"


async def omdb_request(params: dict):
    api_key = os.getenv("OMDB_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="Title search is unavailable: OMDB_API_KEY is not configured.")
    def fetch():
        url = f"{OMDB_URL}?{urlencode({**params, 'apikey': api_key})}"
        with urlopen(url, timeout=8) as response:
            return json.load(response)

    try:
        data = await asyncio.to_thread(fetch)
    except (URLError, TimeoutError) as exc:
        raise HTTPException(status_code=502, detail="OMDb is unavailable. Please try again shortly.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=502, detail="OMDb returned an invalid response.") from exc
    if not isinstance(data, dict):
        raise HTTPException(status_code=502, detail="OMDb returned an invalid response.")
    return data


@app.get("/omdb/search")
async def search_omdb(
    title: str = Query(min_length=1, max_length=100),
    media_type: Literal["movie", "series"] | None = None,
):
    title = title.strip()
    if not title:
        raise HTTPException(status_code=422, detail="Enter a title to search.")
    params = {"s": title}
    if media_type:
        params["type"] = media_type
    data = await omdb_request(params)
    if data.get("Response") == "False":
        if data.get("Error") == "Movie not found!":
            return {"results": [], "total_results": 0}
        raise HTTPException(status_code=502, detail=data.get("Error", "OMDb search failed."))
    return {"results": data.get("Search", []), "total_results": int(data.get("totalResults", 0))}


@app.get("/omdb/title/{imdb_id}")
async def omdb_detail(imdb_id: str = Path(min_length=9, max_length=12)):
    if not re.fullmatch(r"tt\d{7,10}", imdb_id):
        raise HTTPException(status_code=422, detail="Invalid IMDb ID.")
    data = await omdb_request({"i": imdb_id, "plot": "full"})
    if data.get("Response") == "False":
        if data.get("Error") == "Incorrect IMDb ID.":
            raise HTTPException(status_code=404, detail="Title not found.")
        raise HTTPException(status_code=502, detail=data.get("Error", "OMDb detail request failed."))
    return data

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
    if query.tmdb_id is not None:
        matches = data[(data["id"] == query.tmdb_id) & (data["title"].astype(str).str.casefold() == title)].index
    else:
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
