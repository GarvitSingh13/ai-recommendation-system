import re
import numpy as np
import pandas as pd
from collections import Counter


class FeatureExtractor:
    """
    Computes and caches user, movie, and interaction features
    for the Stage 2 LightGBM Re-Ranking Model.
    """
    def __init__(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame):
        self.user_stats = {}
        self.movie_stats = {}
        self._build_stats(ratings_df, movies_df)

    def _extract_year(self, date_str: str) -> int:
        """Extracts 4-digit release year from date strings like '01-Jan-1995'."""
        if not date_str or pd.isna(date_str):
            return 1990
        match = re.search(r"\b(19\d\d|20\d\d)\b", str(date_str))
        return int(match.group(1)) if match else 1990

    def _build_stats(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame):
        print("Precomputing user and movie statistics for ranking features...")
        
        # 1. Movie Statistics (Popularity & Quality)
        movie_grp = ratings_df.groupby("movie_id")
        movie_counts = movie_grp["rating"].count().to_dict()
        movie_means = movie_grp["rating"].mean().to_dict()

        for _, row in movies_df.iterrows():
            mid = int(row["movie_id"])
            genres = set(str(row["genres"]).split("|"))
            year = self._extract_year(row.get("release_date", ""))
            
            self.movie_stats[mid] = {
                "movie_rating_count": movie_counts.get(mid, 0),
                "movie_avg_rating": float(movie_means.get(mid, 3.0)),
                "movie_release_year": year,
                "genres": genres
            }

        # 2. User Statistics (Activity & Favorite Genres)
        user_grp = ratings_df.groupby("user_id")
        user_counts = user_grp["rating"].count().to_dict()
        user_means = user_grp["rating"].mean().to_dict()

        # Find top 3 favorite genres for each user (from movies they rated >= 4.0)
        high_rated = ratings_df[ratings_df["rating"] >= 4.0].merge(movies_df, on="movie_id")
        user_top_genres = {}
        for uid, group in high_rated.groupby("user_id"):
            all_genres = []
            for g_str in group["genres"]:
                all_genres.extend(str(g_str).split("|"))
            top_3 = [g for g, _ in Counter(all_genres).most_common(3)]
            user_top_genres[uid] = set(top_3)

        global_avg_rating = float(ratings_df["rating"].mean())
        for uid in user_counts:
            self.user_stats[uid] = {
                "user_rating_count": user_counts[uid],
                "user_avg_rating": float(user_means[uid]),
                "favorite_genres": user_top_genres.get(uid, set())
            }
        self.global_avg_rating = global_avg_rating
        print(f"Stats precomputed for {len(self.user_stats)} users and {len(self.movie_stats)} movies.")

    def extract_features(self, user_id: int, movie_id: int, vector_sim: float = 0.0) -> dict:
        """Generates the full feature dictionary for a single (user, movie) candidate pair."""
        u_info = self.user_stats.get(user_id, {
            "user_rating_count": 0,
            "user_avg_rating": self.global_avg_rating,
            "favorite_genres": set()
        })
        m_info = self.movie_stats.get(movie_id, {
            "movie_rating_count": 0,
            "movie_avg_rating": self.global_avg_rating,
            "movie_release_year": 1990,
            "genres": set()
        })

        # Cross-features
        overlap_genres = len(u_info["favorite_genres"].intersection(m_info["genres"]))
        rating_diff = m_info["movie_avg_rating"] - u_info["user_avg_rating"]

        return {
            "user_rating_count": u_info["user_rating_count"],
            "user_avg_rating": u_info["user_avg_rating"],
            "movie_rating_count": m_info["movie_rating_count"],
            "movie_avg_rating": m_info["movie_avg_rating"],
            "movie_release_year": m_info["movie_release_year"],
            "rating_diff": rating_diff,
            "genre_overlap_count": overlap_genres,
            "vector_similarity": float(vector_sim)
        }
