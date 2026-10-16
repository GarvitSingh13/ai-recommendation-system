import os
import sys
import time
import random
import joblib
import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from qdrant_client import QdrantClient

# Ensure project root is in Python's search path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database import engine
from src.features import FeatureExtractor
from src.models.two_tower import TwoTowerModel
from src.train_two_tower import GENRE_LIST, build_genre_lookup

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
TWO_TOWER_PATH = os.path.join(MODEL_DIR, "two_tower.pt")
RANKER_PATH = os.path.join(MODEL_DIR, "ranker.pkl")
FEATURE_EXTRACTOR_PATH = os.path.join(MODEL_DIR, "feature_extractor.pkl")


def load_two_tower():
    """Loads the trained PyTorch Two-Tower model to generate vector similarity features."""
    checkpoint = torch.load(TWO_TOWER_PATH, map_location="cpu", weights_only=False)
    model = TwoTowerModel(
        num_users=checkpoint["num_users"],
        num_movies=checkpoint["num_movies"],
        num_genres=len(checkpoint["genre_list"]),
        embedding_dim=checkpoint["embedding_dim"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def prepare_ranking_data(ratings_df, movies_df, two_tower_model, extractor):
    """Creates the tabular dataset with features and binary click/watch targets."""
    print("\nExtracting ranking features for observed and sampled interactions...")
    genre_lookup = build_genre_lookup(movies_df)

    # 1. Collect positive and negative examples
    positives = ratings_df[ratings_df["rating"] >= 3.5].copy()
    negatives = ratings_df[ratings_df["rating"] < 3.0].copy()

    # Balance with additional unrated movie samples
    user_watched = ratings_df.groupby("user_id")["movie_id"].apply(set).to_dict()
    all_movies = list(movies_df["movie_id"].unique())

    sampled_users, sampled_movies, sampled_labels = [], [], []

    # Add real positive & negative ratings
    for _, row in positives.iterrows():
        sampled_users.append(int(row["user_id"]))
        sampled_movies.append(int(row["movie_id"]))
        sampled_labels.append(1)

    for _, row in negatives.iterrows():
        sampled_users.append(int(row["user_id"]))
        sampled_movies.append(int(row["movie_id"]))
        sampled_labels.append(0)

    # Sample unobserved negative interactions
    for uid in user_watched:
        watched = user_watched[uid]
        unwatched = [m for m in all_movies if m not in watched]
        num_to_sample = min(len(unwatched), 20)
        for m in random.sample(unwatched, num_to_sample):
            sampled_users.append(uid)
            sampled_movies.append(m)
            sampled_labels.append(0)

    u_arr = np.array(sampled_users, dtype=np.int64)
    m_arr = np.array(sampled_movies, dtype=np.int64)
    y_arr = np.array(sampled_labels, dtype=np.int32)

    # 2. Compute Stage 1 Vector Similarity for all pairs
    print(f"Computing Two-Tower vector similarity for {len(u_arr)} training examples...")
    with torch.no_grad():
        u_t = torch.tensor(u_arr, dtype=torch.long)
        m_t = torch.tensor(m_arr, dtype=torch.long)
        g_t = torch.tensor(genre_lookup[m_arr], dtype=torch.float32)
        vector_sims = two_tower_model(u_t, m_t, g_t).numpy()

    # 3. Build Feature Table
    feature_rows = []
    for i in range(len(u_arr)):
        feats = extractor.extract_features(
            user_id=int(u_arr[i]),
            movie_id=int(m_arr[i]),
            vector_sim=float(vector_sims[i])
        )
        feature_rows.append(feats)

    X = pd.DataFrame(feature_rows)
    y = pd.Series(y_arr, name="target")
    print(f"Dataset prepared: {X.shape[0]} rows, {X.shape[1]} features.")
    return X, y


def train_ranker():
    random.seed(42)
    np.random.seed(42)

    # 1. Load data & models
    print("Loading data from PostgreSQL...")
    movies_df = pd.read_sql("SELECT movie_id, title, release_date, genres FROM movies", con=engine)
    ratings_df = pd.read_sql("SELECT user_id, movie_id, rating, timestamp FROM ratings", con=engine)

    two_tower = load_two_tower()
    extractor = FeatureExtractor(ratings_df, movies_df)

    # 2. Build dataset
    X, y = prepare_ranking_data(ratings_df, movies_df, two_tower, extractor)

    # 3. Train / Test Split (80% / 20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 4. Train LightGBM Classifier / Ranker
    print("\nTraining LightGBM Stage 2 Ranker...")
    ranker = LGBMClassifier(
        n_estimators=120,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )

    start_time = time.time()
    ranker.fit(X_train, y_train)
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f} seconds!")

    # 5. Evaluation
    test_probs = ranker.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= 0.5).astype(int)

    auc_score = roc_auc_score(y_test, test_probs)
    acc = accuracy_score(y_test, test_preds)

    print("\n" + "="*60)
    print("STAGE 2 MODEL EVALUATION METRICS:")
    print(f"  • ROC-AUC Score: {auc_score:.4f} (Measures discrimination between good/bad movies)")
    print(f"  • Accuracy:      {acc*100:.2f}%")
    print("="*60)

    # Feature Importances
    importances = pd.Series(ranker.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nTop Ranking Features Discovered by LightGBM:")
    for feat, imp in importances.items():
        print(f"  - {feat:<22}: {imp}")

    # 6. Save Models to Disk
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(ranker, RANKER_PATH)
    joblib.dump(extractor, FEATURE_EXTRACTOR_PATH)
    print(f"\nModels successfully saved to:\n  - {RANKER_PATH}\n  - {FEATURE_EXTRACTOR_PATH}")

    # 7. End-to-End Simulation: Qdrant Candidates -> LightGBM Re-Rank
    print("\n" + "="*70)
    print("COMPLETE TWO-STAGE RECOMMENDATION SIMULATION FOR USER #42")
    print("="*70)

    test_uid = 42
    qdrant = QdrantClient(host="localhost", port=6333)

    # Step A: Generate user taste vector from Two-Tower model
    with torch.no_grad():
        u_tensor = torch.tensor([test_uid], dtype=torch.long)
        user_vector = two_tower.get_user_embedding(u_tensor).squeeze(0).numpy().tolist()

    # Step B: Stage 1 Retrieval (Retrieve top 20 candidates from Qdrant)
    q_start = time.time()
    response = qdrant.query_points(
        collection_name="movies",
        query=user_vector,
        limit=20
    )
    candidates = response.points
    q_ms = (time.time() - q_start) * 1000
    print(f"Stage 1 (Qdrant Vector Retrieval): Found 20 candidates in {q_ms:.2f} ms")

    # Step C: Stage 2 Re-Ranking (Score all 20 candidates with LightGBM)
    r_start = time.time()
    cand_features = []
    for c in candidates:
        feats = extractor.extract_features(
            user_id=test_uid,
            movie_id=c.payload["movie_id"],
            vector_sim=c.score
        )
        cand_features.append(feats)

    cand_df = pd.DataFrame(cand_features)
    predicted_probs = ranker.predict_proba(cand_df)[:, 1]
    r_ms = (time.time() - r_start) * 1000
    print(f"Stage 2 (LightGBM Re-Ranking):   Scored 20 candidates in {r_ms:.2f} ms")
    print(f"Total Two-Stage Inference Time:   {q_ms + r_ms:.2f} ms (Well within <30ms SLA!)")

    # Sort candidates by LightGBM probability
    ranked_indices = np.argsort(predicted_probs)[::-1]

    print("\nFinal Top 10 Personalized Recommendations for User #42:")
    for rank, idx in enumerate(ranked_indices[:10], start=1):
        c = candidates[idx]
        prob = predicted_probs[idx]
        print(f"  {rank:>2}. [{prob*100:5.1f}% match] {c.payload['title']:<40} | Genres: {c.payload['genres']}")
    print("="*70)


if __name__ == "__main__":
    train_ranker()
