import os
import sys
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Ensure project root is in Python's search path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database import engine
from src.models.two_tower import TwoTowerModel

GENRE_LIST = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]


class RecsysDataset(Dataset):
    """PyTorch Dataset for user-movie interaction pairs."""
    def __init__(self, user_ids, movie_ids, genre_matrix, labels):
        self.user_ids = torch.tensor(user_ids, dtype=torch.long)
        self.movie_ids = torch.tensor(movie_ids, dtype=torch.long)
        self.genre_vectors = torch.tensor(genre_matrix, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.user_ids[idx],
            self.movie_ids[idx],
            self.genre_vectors[idx],
            self.labels[idx]
        )


def load_data_from_postgres():
    """Fetches movies and ratings directly from the PostgreSQL container."""
    print("Loading movies and interaction data from PostgreSQL...")
    movies_df = pd.read_sql("SELECT movie_id, title, genres FROM movies", con=engine)
    ratings_df = pd.read_sql("SELECT user_id, movie_id, rating, timestamp FROM ratings", con=engine)
    print(f"Loaded {len(movies_df)} movies and {len(ratings_df)} user ratings.")
    return movies_df, ratings_df


def build_genre_lookup(movies_df):
    """Builds a lookup table mapping movie_id -> 19-dimensional multi-hot genre array."""
    genre_to_idx = {g: i for i, g in enumerate(GENRE_LIST)}
    max_movie_id = movies_df["movie_id"].max()
    genre_matrix = np.zeros((max_movie_id + 1, len(GENRE_LIST)), dtype=np.float32)

    for _, row in movies_df.iterrows():
        mid = row["movie_id"]
        genres = str(row["genres"]).split("|")
        for g in genres:
            if g in genre_to_idx:
                genre_matrix[mid, genre_to_idx[g]] = 1.0

    return genre_matrix


def prepare_training_pairs(ratings_df, all_movie_ids, negative_ratio=2):
    """
    Creates positive interaction pairs (rating >= 3.5) and samples
    unobserved negative pairs for candidate retrieval training.
    """
    print("\nGenerating positive pairs and sampling negative interactions...")
    # Positive interactions: User liked the movie (rating >= 3.5)
    positives = ratings_df[ratings_df["rating"] >= 3.5].copy()
    user_watched = ratings_df.groupby("user_id")["movie_id"].apply(set).to_dict()

    user_ids, movie_ids, labels = [], [], []

    # 1. Add all positive pairs
    for _, row in positives.iterrows():
        user_ids.append(row["user_id"])
        movie_ids.append(row["movie_id"])
        labels.append(1.0)

    # 2. Add sampled negative pairs (unrated movies)
    movie_pool = list(all_movie_ids)
    for u in user_watched:
        watched = user_watched[u]
        unwatched = [m for m in movie_pool if m not in watched]
        num_neg = min(len(unwatched), int(len(watched) * negative_ratio))
        neg_samples = random.sample(unwatched, num_neg) if unwatched else []
        for m in neg_samples:
            user_ids.append(u)
            movie_ids.append(m)
            labels.append(0.0)

    print(f"Total training interaction pairs: {len(labels)} ({sum(labels):.0f} positive, {len(labels)-sum(labels):.0f} negative)")
    return (
        np.array(user_ids, dtype=np.int64),
        np.array(movie_ids, dtype=np.int64),
        np.array(labels, dtype=np.float32)
    )


def train_model():
    # Set reproducible seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # 1. Load data
    movies_df, ratings_df = load_data_from_postgres()
    genre_lookup = build_genre_lookup(movies_df)

    num_users = int(ratings_df["user_id"].max())
    num_movies = int(movies_df["movie_id"].max())
    all_movie_ids = set(movies_df["movie_id"].unique())

    # 2. Build dataset
    u_ids, m_ids, labels = prepare_training_pairs(ratings_df, all_movie_ids, negative_ratio=2)
    genres = genre_lookup[m_ids]

    # 3. Train / Test Split (80% / 20%)
    train_u, test_u, train_m, test_m, train_g, test_g, train_y, test_y = train_test_split(
        u_ids, m_ids, genres, labels, test_size=0.20, random_state=42, stratify=labels
    )

    train_dataset = RecsysDataset(train_u, train_m, train_g, train_y)
    test_dataset = RecsysDataset(test_u, test_m, test_g, test_y)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    # 4. Initialize Two-Tower Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining on device: {device}")
    
    model = TwoTowerModel(
        num_users=num_users,
        num_movies=num_movies,
        num_genres=len(GENRE_LIST),
        embedding_dim=64
    ).to(device)

    # Cosine temperature scaling: cosine similarity in [-1, 1] scaled by 10 to form logits
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)

    # 5. Training Loop
    epochs = 5
    print("\nStarting training (5 epochs)...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for u, m, g, y in train_loader:
            u, m, g, y = u.to(device), m.to(device), g.to(device), y.to(device)
            optimizer.zero_grad()

            similarity = model(u, m, g)
            # Logits for BCE
            logits = similarity * 10.0
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)

        avg_loss = total_loss / len(train_dataset)

        # Validation evaluation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for u, m, g, y in test_loader:
                u, m, g, y = u.to(device), m.to(device), g.to(device), y.to(device)
                similarity = model(u, m, g)
                preds = (similarity > 0.0).float()
                correct += (preds == y).sum().item()
                total += len(y)

        val_acc = (correct / total) * 100
        print(f"Epoch [{epoch}/{epochs}] - Loss: {avg_loss:.4f} | Validation Accuracy: {val_acc:.2f}%")

    # 6. Recommendation Metric: HitRate@10 on Test Set
    print("\nEvaluating Recommendation Metric: HitRate@10...")
    model.eval()
    test_positives = [(u, m) for u, m, y in zip(test_u, test_m, test_y) if y == 1.0]
    sample_eval = random.sample(test_positives, min(1000, len(test_positives)))

    # Precompute all movie embeddings
    all_m = torch.tensor(movies_df["movie_id"].values.astype(np.int64), dtype=torch.long).to(device)
    all_g = torch.tensor(genre_lookup[movies_df["movie_id"].values.astype(np.int64)], dtype=torch.float32).to(device)
    with torch.no_grad():
        all_movie_embeddings = model.get_movie_embedding(all_m, all_g)

    hits_at_10 = 0
    hits_at_50 = 0
    movie_id_map = movies_df["movie_id"].values

    with torch.no_grad():
        for u_id, target_m in sample_eval:
            user_t = torch.tensor([u_id], dtype=torch.long).to(device)
            user_emb = model.get_user_embedding(user_t) # (1, 64)
            scores = torch.sum(user_emb * all_movie_embeddings, dim=-1).cpu().numpy()
            
            top_indices = np.argsort(scores)[::-1]
            top_10 = movie_id_map[top_indices[:10]]
            top_50 = movie_id_map[top_indices[:50]]

            if target_m in top_10:
                hits_at_10 += 1
            if target_m in top_50:
                hits_at_50 += 1

    hr_10 = (hits_at_10 / len(sample_eval)) * 100
    hr_50 = (hits_at_50 / len(sample_eval)) * 100
    print(f"HitRate@10: {hr_10:.2f}% (Out of 1,682 movies, the true liked movie was in Top 10!)")
    print(f"HitRate@50: {hr_50:.2f}% (Found in Top 50 candidates for Stage 2 re-ranking)")

    # 7. Save model checkpoint
    os.makedirs(os.path.join(os.path.dirname(os.path.dirname(__file__)), "models"), exist_ok=True)
    model_save_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "two_tower.pt")
    
    torch.save({
        "model_state_dict": model.state_dict(),
        "num_users": num_users,
        "num_movies": num_movies,
        "embedding_dim": 64,
        "genre_list": GENRE_LIST
    }, model_save_path)
    print(f"\nModel successfully saved to {model_save_path}!")


if __name__ == "__main__":
    train_model()
