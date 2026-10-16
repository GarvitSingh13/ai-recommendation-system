import os
import sys
import time
import numpy as np
import pandas as pd
import torch
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Ensure project root is in Python's search path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database import engine
from src.models.two_tower import TwoTowerModel
from src.train_two_tower import GENRE_LIST, build_genre_lookup

COLLECTION_NAME = "movies"
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "two_tower.pt")


def load_trained_model():
    """Loads the trained Two-Tower model checkpoint."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}. Please run src/train_two_tower.py first.")

    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model = TwoTowerModel(
        num_users=checkpoint["num_users"],
        num_movies=checkpoint["num_movies"],
        num_genres=len(checkpoint["genre_list"]),
        embedding_dim=checkpoint["embedding_dim"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("Trained Two-Tower model loaded successfully!")
    return model


def index_movies_to_qdrant():
    # 1. Connect to Qdrant container
    print(f"Connecting to Qdrant Vector Database at {QDRANT_HOST}:{QDRANT_PORT}...")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # 2. Re-create collection cleanly
    print(f"Setting up collection '{COLLECTION_NAME}' (64-dim, Cosine distance)...")
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=64, distance=models.Distance.COSINE)
    )

    # 3. Load movies and generate embeddings
    model = load_trained_model()
    movies_df = pd.read_sql("SELECT movie_id, title, release_date, genres FROM movies", con=engine)
    genre_lookup = build_genre_lookup(movies_df)

    movie_ids = movies_df["movie_id"].values.astype(np.int64)
    m_tensor = torch.tensor(movie_ids, dtype=torch.long)
    g_tensor = torch.tensor(genre_lookup[movie_ids], dtype=torch.float32)

    print("Generating 64-dimensional embeddings for all 1,682 movies using Movie Tower...")
    with torch.no_grad():
        movie_embeddings = model.get_movie_embedding(m_tensor, g_tensor).numpy()

    # 4. Upload to Qdrant in batch
    print(f"Uploading {len(movies_df)} movie vectors into Qdrant...")
    points = []
    for idx, row in movies_df.iterrows():
        mid = int(row["movie_id"])
        vec = movie_embeddings[idx].tolist()
        payload = {
            "movie_id": mid,
            "title": str(row["title"]),
            "genres": str(row["genres"]),
            "release_date": str(row["release_date"])
        }
        points.append(models.PointStruct(id=mid, vector=vec, payload=payload))

    # Upload in chunks of 500
    chunk_size = 500
    for i in range(0, len(points), chunk_size):
        client.upsert(collection_name=COLLECTION_NAME, points=points[i:i + chunk_size])

    print("All movie vectors successfully indexed in Qdrant!")

    # 5. Live Test Search Simulation
    print("\n" + "="*60)
    print("LIVE RETRIEVAL TEST: Testing Sub-5ms Candidate Search")
    print("="*60)

    test_user_id = 42 # Sample user
    with torch.no_grad():
        user_tensor = torch.tensor([test_user_id], dtype=torch.long)
        user_vector = model.get_user_embedding(user_tensor).squeeze(0).numpy().tolist()

    start_time = time.time()
    # Use modern Qdrant query_points API
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=user_vector,
        limit=5
    )
    search_results = response.points
    elapsed_ms = (time.time() - start_time) * 1000

    print(f"Candidate Retrieval for User #{test_user_id} took: {elapsed_ms:.2f} milliseconds!")
    print(f"\nTop 5 Retrieved Movie Candidates:")
    for rank, hit in enumerate(search_results, start=1):
        print(f"  {rank}. [{hit.score:.4f} match] {hit.payload['title']} ({hit.payload['genres']})")
    print("="*60)


if __name__ == "__main__":
    index_movies_to_qdrant()
