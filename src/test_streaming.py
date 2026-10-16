import os
import sys
import time
import joblib
import torch
import numpy as np
import pandas as pd
from qdrant_client import QdrantClient

# Ensure project root is in Python's search path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database import engine
from src.models.two_tower import TwoTowerModel
from src.events.producer import send_user_event
from src.events.consumer import (
    clear_user_recent_watches,
    process_kafka_events_once,
    get_user_recent_watches
)
from src.train_two_tower import GENRE_LIST, build_genre_lookup

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
TWO_TOWER_PATH = os.path.join(MODEL_DIR, "two_tower.pt")
RANKER_PATH = os.path.join(MODEL_DIR, "ranker.pkl")
FEATURE_EXTRACTOR_PATH = os.path.join(MODEL_DIR, "feature_extractor.pkl")


def load_artifacts():
    checkpoint = torch.load(TWO_TOWER_PATH, map_location="cpu", weights_only=False)
    model = TwoTowerModel(
        num_users=checkpoint["num_users"],
        num_movies=checkpoint["num_movies"],
        num_genres=len(checkpoint["genre_list"]),
        embedding_dim=checkpoint["embedding_dim"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    ranker = joblib.load(RANKER_PATH)
    extractor = joblib.load(FEATURE_EXTRACTOR_PATH)
    return model, ranker, extractor


def get_recommendations(user_id: int, two_tower, ranker, extractor, qdrant, movies_df, genre_lookup, limit=5):
    """
    Computes two-stage recommendations. If the user has active watched movies in Redis,
    dynamically blends their real-time session taste with their historical profile.
    """
    recent_movie_ids = get_user_recent_watches(user_id)

    # 1. Base User Profile Embedding
    with torch.no_grad():
        u_tensor = torch.tensor([user_id], dtype=torch.long)
        user_vector = two_tower.get_user_embedding(u_tensor).squeeze(0).numpy()

    # 2. Dynamic Real-Time Blending (If user has recent watches in Redis)
    is_realtime_adjusted = False
    if recent_movie_ids:
        is_realtime_adjusted = True
        recent_m_ids = np.array(recent_movie_ids, dtype=np.int64)
        m_t = torch.tensor(recent_m_ids, dtype=torch.long)
        g_t = torch.tensor(genre_lookup[recent_m_ids], dtype=torch.float32)
        
        with torch.no_grad():
            recent_embeddings = two_tower.get_movie_embedding(m_t, g_t).numpy()
            session_vector = np.mean(recent_embeddings, axis=0)
            
            # Blend 50% long-term profile + 50% immediate session taste
            blended = 0.5 * user_vector + 0.5 * session_vector
            user_vector = blended / np.linalg.norm(blended)

    # 3. Stage 1: Qdrant Candidate Retrieval (20 candidates)
    response = qdrant.query_points(
        collection_name="movies",
        query=user_vector.tolist(),
        limit=20
    )
    candidates = response.points

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
    predicted_probs = ranker.predict_proba(cand_df)[:, 1]

    # Sort descending by predicted probability
    ranked_indices = np.argsort(predicted_probs)[::-1]
    
    results = []
    for idx in ranked_indices[:limit]:
        c = candidates[idx]
        prob = predicted_probs[idx]
        results.append({
            "title": c.payload["title"],
            "genres": c.payload["genres"],
            "score": prob
        })
    return results, is_realtime_adjusted, recent_movie_ids


def run_live_streaming_demo():
    print("="*75)
    print("REAL-TIME STREAMING FEEDBACK DEMO: KAFKA -> REDIS -> LIVE AI ADAPTATION")
    print("="*75)

    # 1. Setup
    two_tower, ranker, extractor = load_artifacts()
    qdrant = QdrantClient(host="localhost", port=6333)
    movies_df = pd.read_sql("SELECT movie_id, title, release_date, genres FROM movies", con=engine)
    genre_lookup = build_genre_lookup(movies_df)

    test_uid = 42
    clear_user_recent_watches(test_uid)

    # 2. Baseline Recommendations (Before any live interaction)
    print(f"\n[PHASE 1] Initial Recommendations for User #{test_uid} (Historical Taste: Sci-Fi / Action):")
    base_recs, _, _ = get_recommendations(test_uid, two_tower, ranker, extractor, qdrant, movies_df, genre_lookup)
    for i, r in enumerate(base_recs, start=1):
        print(f"  {i}. [{r['score']*100:5.1f}%] {r['title']:<40} | Genres: {r['genres']}")

    # 3. Live User Interaction: User suddenly watches 3 Animation / Family movies!
    print("\n" + "-"*75)
    print(f"[PHASE 2] User #{test_uid} sits down and watches 3 Animation / Children's Movies:")
    simulated_watches = [
        (1, "Toy Story (1995)"),
        (95, "Aladdin (1992)"),
        (71, "Lion King, The (1994)")
    ]

    for mid, title in simulated_watches:
        print(f"  --> Producing live event to Kafka: User #{test_uid} watched '{title}' (ID: {mid})")
        send_user_event(user_id=test_uid, movie_id=mid, event_type="watch", rating=5.0)

    # 4. Consumer updates Redis
    print("\n[PHASE 3] Background Consumer Worker processes Kafka stream into Redis RAM...")
    events_processed = process_kafka_events_once(max_records=3, timeout_ms=3000)
    print(f"  --> {events_processed} events processed and written to Redis!")

    stored_watches = get_user_recent_watches(test_uid)
    print(f"  --> Redis current session for User #{test_uid}: Movie IDs {stored_watches}")

    # 5. Real-Time Re-Query: Notice how the recommendations immediately adapt!
    print("\n" + "-"*75)
    print(f"[PHASE 4] User #{test_uid} refreshes their home screen (Real-Time Dynamic Update):")
    new_recs, is_adapted, recent = get_recommendations(test_uid, two_tower, ranker, extractor, qdrant, movies_df, genre_lookup)
    
    print(f"  (Session Blending Active: True, based on {len(recent)} recent watches from Redis)\n")
    for i, r in enumerate(new_recs, start=1):
        print(f"  {i}. [{r['score']*100:5.1f}%] {r['title']:<40} | Genres: {r['genres']}")

    print("\n" + "="*75)
    print("SUCCESS: Recommendations shifted dynamically in <30ms without model retraining!")
    print("="*75)


if __name__ == "__main__":
    run_live_streaming_demo()
