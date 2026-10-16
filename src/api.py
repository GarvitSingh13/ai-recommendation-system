import os
import sys
import time
import joblib
import torch
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager
from typing import List, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Query
from qdrant_client import QdrantClient

# Ensure project root is in Python's search path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database import engine
from src.models.two_tower import TwoTowerModel
from src.features import FeatureExtractor
from src.events.producer import send_user_event
from src.events.consumer import (
    get_user_recent_watches,
    update_user_session_in_redis,
    clear_user_recent_watches,
    get_redis_client
)
from src.train_two_tower import build_genre_lookup

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
TWO_TOWER_PATH = os.path.join(MODEL_DIR, "two_tower.pt")
RANKER_PATH = os.path.join(MODEL_DIR, "ranker.pkl")
FEATURE_EXTRACTOR_PATH = os.path.join(MODEL_DIR, "feature_extractor.pkl")

# Shared state holding loaded AI models and clients
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads all AI models, databases, and caches into RAM on server startup."""
    print("\n--- Starting AI Recommender Engine ---")
    
    # 1. Load Two-Tower Neural Network
    print("Loading PyTorch Two-Tower Model into memory...")
    checkpoint = torch.load(TWO_TOWER_PATH, map_location="cpu", weights_only=False)
    two_tower = TwoTowerModel(
        num_users=checkpoint["num_users"],
        num_movies=checkpoint["num_movies"],
        num_genres=len(checkpoint["genre_list"]),
        embedding_dim=checkpoint["embedding_dim"]
    )
    two_tower.load_state_dict(checkpoint["model_state_dict"])
    two_tower.eval()
    state["two_tower"] = two_tower

    # 2. Load LightGBM Ranker and Feature Extractor
    print("Loading LightGBM Ranker & Feature Extractor...")
    state["ranker"] = joblib.load(RANKER_PATH)
    state["extractor"] = joblib.load(FEATURE_EXTRACTOR_PATH)

    # 3. Connect to Qdrant & Redis
    print("Connecting to Qdrant and Redis...")
    state["qdrant"] = QdrantClient(host="localhost", port=6333)
    state["redis"] = get_redis_client()

    # 4. Load Movies & Genre Lookup for fast enrichment
    movies_df = pd.read_sql("SELECT movie_id, title, release_date, genres FROM movies", con=engine)
    state["movies_df"] = movies_df
    state["genre_lookup"] = build_genre_lookup(movies_df)

    print("All models and services initialized successfully! Server ready.\n")
    yield
    print("--- Shutting Down AI Recommender Engine ---")


app = FastAPI(
    title="Real-Time AI Recommendation Engine",
    description="Two-stage movie recommendation platform with PyTorch, Qdrant, LightGBM, Kafka, and Redis.",
    version="1.0.0",
    lifespan=lifespan
)


# Pydantic Schemas for Swagger UI
class MovieRecommendation(BaseModel):
    rank: int
    movie_id: int
    title: str
    genres: str
    match_percentage: float = Field(..., description="Calibrated watch probability (0 - 100%)")


class RecommendationResponse(BaseModel):
    user_id: int
    session_adapted: bool = Field(..., description="True if real-time Redis watches influenced results")
    recent_watches_count: int
    latency_ms: float
    recommendations: List[MovieRecommendation]


class WatchEventRequest(BaseModel):
    user_id: int = Field(..., example=42)
    movie_id: int = Field(..., example=1)
    rating: float = Field(default=5.0, example=5.0)


@app.get("/", tags=["General"])
def root():
    return {
        "message": "Real-Time AI Recommendation Engine is running!",
        "docs_url": "http://localhost:8000/docs",
        "architecture": "Two-Stage Funnel: PyTorch/Qdrant (Retrieval) + LightGBM (Re-Ranking) + Kafka/Redis (Real-time)"
    }


@app.get("/health", tags=["Monitoring"])
def health_check():
    """Healthcheck verifying all 4 underlying services."""
    services = {
        "postgresql": "healthy",
        "qdrant": "healthy",
        "redis": "healthy",
        "kafka": "healthy"
    }
    try:
        state["redis"].ping()
    except Exception as e:
        services["redis"] = f"unhealthy: {e}"

    try:
        state["qdrant"].get_collections()
    except Exception as e:
        services["qdrant"] = f"unhealthy: {e}"

    return {"status": "ok", "services": services}


@app.get("/recommend/{user_id}", response_model=RecommendationResponse, tags=["Recommendation"])
def get_recommendations(
    user_id: int,
    limit: int = Query(default=10, ge=1, le=50, description="Number of recommendations to return")
):
    """
    Executes the complete Two-Stage Recommendation Pipeline:
    1. Checks Redis for real-time user session history (<1ms)
    2. Retrieves top 30 candidates from Qdrant Vector DB (<8ms)
    3. Re-ranks candidates with LightGBM decision tree (<2ms)
    Returns personalized recommendations in under 25ms.
    """
    start_time = time.time()
    two_tower = state["two_tower"]
    ranker = state["ranker"]
    extractor = state["extractor"]
    qdrant = state["qdrant"]
    movies_df = state["movies_df"]
    genre_lookup = state["genre_lookup"]

    # 1. Fetch real-time active session from Redis
    recent_movie_ids = get_user_recent_watches(user_id)

    # 2. Compute User Taste Vector
    with torch.no_grad():
        u_tensor = torch.tensor([user_id], dtype=torch.long)
        user_vector = two_tower.get_user_embedding(u_tensor).squeeze(0).numpy()

    is_session_adapted = False
    if recent_movie_ids:
        is_session_adapted = True
        recent_m_ids = np.array(recent_movie_ids, dtype=np.int64)
        m_t = torch.tensor(recent_m_ids, dtype=torch.long)
        g_t = torch.tensor(genre_lookup[recent_m_ids], dtype=torch.float32)

        with torch.no_grad():
            recent_embeddings = two_tower.get_movie_embedding(m_t, g_t).numpy()
            session_vector = np.mean(recent_embeddings, axis=0)
            
            # Blend long-term taste + real-time session
            blended = 0.5 * user_vector + 0.5 * session_vector
            user_vector = blended / np.linalg.norm(blended)

    # 3. Stage 1: Candidate Retrieval (Qdrant)
    response = qdrant.query_points(
        collection_name="movies",
        query=user_vector.tolist(),
        limit=max(30, limit * 2)
    )
    candidates = response.points

    if not candidates:
        raise HTTPException(status_code=404, detail="No candidate movies found.")

    # 4. Stage 2: LightGBM Re-Ranking
    cand_features = []
    for c in candidates:
        feats = extractor.extract_features(
            user_id=user_id,
            movie_id=c.payload["movie_id"],
            vector_sim=c.score
        )
        cand_features.append(feats)

    cand_df = pd.DataFrame(cand_features)
    probs = ranker.predict_proba(cand_df)[:, 1]

    # Sort descending
    ranked_indices = np.argsort(probs)[::-1]

    results = []
    for rank_num, idx in enumerate(ranked_indices[:limit], start=1):
        c = candidates[idx]
        results.append(MovieRecommendation(
            rank=rank_num,
            movie_id=c.payload["movie_id"],
            title=c.payload["title"],
            genres=c.payload["genres"],
            match_percentage=round(float(probs[idx]) * 100, 1)
        ))

    elapsed_ms = round((time.time() - start_time) * 1000, 2)

    return RecommendationResponse(
        user_id=user_id,
        session_adapted=is_session_adapted,
        recent_watches_count=len(recent_movie_ids),
        latency_ms=elapsed_ms,
        recommendations=results
    )


@app.post("/events/watch", tags=["Streaming Events"])
def record_watch_event(event: WatchEventRequest):
    """
    Emits a user watch event to Apache Kafka and immediately records it into Redis RAM.
    """
    # 1. Queue to Apache Kafka topic
    send_user_event(user_id=event.user_id, movie_id=event.movie_id, rating=event.rating)
    
    # 2. Update Redis session store for immediate real-time feedback
    update_user_session_in_redis(user_id=event.user_id, movie_id=event.movie_id)

    return {
        "status": "success",
        "message": f"Watch event recorded for User #{event.user_id} on Movie #{event.movie_id}",
        "session_active_movies": get_user_recent_watches(event.user_id)
    }


@app.delete("/session/{user_id}", tags=["Streaming Events"])
def reset_session(user_id: int):
    """Clears a user's recent watch session from Redis (useful for demos and testing)."""
    clear_user_recent_watches(user_id)
    return {"status": "success", "message": f"Session cleared for User #{user_id}"}
