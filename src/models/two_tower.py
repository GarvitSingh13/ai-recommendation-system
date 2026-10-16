import torch
import torch.nn as nn
import torch.nn.functional as F


class UserTower(nn.Module):
    """Encodes a user into a normalized 64-dimensional embedding vector."""
    def __init__(self, num_users: int, embedding_dim: int = 64):
        super(UserTower, self).__init__()
        self.user_embedding = nn.Embedding(num_users + 1, embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, user_ids: torch.Tensor) -> torch.Tensor:
        x = self.user_embedding(user_ids)
        x = self.mlp(x)
        return F.normalize(x, p=2, dim=-1)


class MovieTower(nn.Module):
    """Encodes a movie (ID + genre features) into a normalized 64-dimensional embedding vector."""
    def __init__(self, num_movies: int, num_genres: int = 19, embedding_dim: int = 64):
        super(MovieTower, self).__init__()
        self.movie_embedding = nn.Embedding(num_movies + 1, embedding_dim)
        self.genre_projection = nn.Linear(num_genres, 32)
        
        # Combine movie embedding (64) + genre projection (32) = 96
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim + 32, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, movie_ids: torch.Tensor, genre_vectors: torch.Tensor) -> torch.Tensor:
        m_emb = self.movie_embedding(movie_ids)
        g_emb = self.genre_projection(genre_vectors)
        combined = torch.cat([m_emb, g_emb], dim=-1)
        x = self.mlp(combined)
        return F.normalize(x, p=2, dim=-1)


class TwoTowerModel(nn.Module):
    """
    Two-Tower Recommendation Model:
    Matches User Taste Vector with Movie Vibe Vector via Dot Product (Cosine Similarity).
    """
    def __init__(self, num_users: int, num_movies: int, num_genres: int = 19, embedding_dim: int = 64):
        super(TwoTowerModel, self).__init__()
        self.user_tower = UserTower(num_users, embedding_dim)
        self.movie_tower = MovieTower(num_movies, num_genres, embedding_dim)

    def forward(self, user_ids: torch.Tensor, movie_ids: torch.Tensor, genre_vectors: torch.Tensor) -> torch.Tensor:
        user_vec = self.user_tower(user_ids)
        movie_vec = self.movie_tower(movie_ids, genre_vectors)
        # Dot product between unit vectors equals cosine similarity in range [-1.0, 1.0]
        similarity = torch.sum(user_vec * movie_vec, dim=-1)
        return similarity

    def get_user_embedding(self, user_ids: torch.Tensor) -> torch.Tensor:
        """Helper to get 64-dim user vectors for online inference."""
        return self.user_tower(user_ids)

    def get_movie_embedding(self, movie_ids: torch.Tensor, genre_vectors: torch.Tensor) -> torch.Tensor:
        """Helper to get 64-dim movie vectors for vector database indexing."""
        return self.movie_tower(movie_ids, genre_vectors)
