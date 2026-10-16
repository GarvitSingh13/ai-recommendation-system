import os
import sys
import zipfile

# Ensure project root is in Python's search path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import pandas as pd
from sqlalchemy import text
from src.database import engine, init_db

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
ZIP_PATH = os.path.join(DATA_DIR, "ml-100k.zip")
EXTRACT_DIR = os.path.join(DATA_DIR, "ml-100k")

# 19 genres in MovieLens 100K
GENRE_COLUMNS = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]


def download_and_extract():
    """Downloads MovieLens-100K dataset if not already present."""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    if not os.path.exists(ZIP_PATH):
        print(f"Downloading MovieLens-100K dataset from {MOVIELENS_URL}...")
        response = requests.get(MOVIELENS_URL, stream=True)
        response.raise_for_status()
        with open(ZIP_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete!")
    else:
        print("Found existing MovieLens zip archive.")

    if not os.path.exists(EXTRACT_DIR):
        print("Extracting dataset...")
        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(DATA_DIR)
        print("Extracted successfully!")
    else:
        print("Dataset already extracted.")


def parse_and_clean_data():
    """Reads raw MovieLens files and returns clean pandas DataFrames."""
    print("\nReading and parsing raw files...")

    # 1. Parse Users: user_id | age | gender | occupation | zip_code
    users_file = os.path.join(EXTRACT_DIR, "u.user")
    users_df = pd.read_csv(
        users_file,
        sep="|",
        names=["user_id", "age", "gender", "occupation", "zip_code"],
        encoding="latin-1"
    )

    # 2. Parse Movies: movie_id | title | release_date | ... | 19 genre flags
    movies_file = os.path.join(EXTRACT_DIR, "u.item")
    item_cols = [
        "movie_id", "title", "release_date", "video_release_date", "imdb_url"
    ] + GENRE_COLUMNS
    
    raw_movies = pd.read_csv(
        movies_file,
        sep="|",
        names=item_cols,
        encoding="latin-1"
    )

    # Combine 19 binary genre columns into a clean string like "Action|Sci-Fi"
    def get_genre_string(row):
        active = [g for g in GENRE_COLUMNS if row[g] == 1 and g != "unknown"]
        return "|".join(active) if active else "General"

    raw_movies["genres"] = raw_movies.apply(get_genre_string, axis=1)
    movies_df = raw_movies[["movie_id", "title", "release_date", "genres"]].copy()

    # 3. Parse Ratings: user_id | movie_id | rating | timestamp
    ratings_file = os.path.join(EXTRACT_DIR, "u.data")
    ratings_df = pd.read_csv(
        ratings_file,
        sep="\t",
        names=["user_id", "movie_id", "rating", "timestamp"]
    )

    print(f"Parsed {len(users_df)} users.")
    print(f"Parsed {len(movies_df)} movies.")
    print(f"Parsed {len(ratings_df)} ratings.")

    return users_df, movies_df, ratings_df


def ingest_to_postgres(users_df, movies_df, ratings_df):
    """Inserts clean DataFrames into PostgreSQL database."""
    print("\nConnecting to PostgreSQL and creating tables...")
    init_db()

    with engine.connect() as conn:
        # Check if data is already loaded
        existing_ratings = conn.execute(text("SELECT COUNT(*) FROM ratings")).scalar()
        if existing_ratings > 0:
            print(f"Notice: 'ratings' table already contains {existing_ratings} records.")
            print("Clearing old data to ensure a fresh, clean ingestion...")
            conn.execute(text("TRUNCATE TABLE ratings, movies, users CASCADE"))
            conn.commit()

    print("Inserting users into PostgreSQL...")
    users_df.to_sql("users", con=engine, if_exists="append", index=False, chunksize=1000)

    print("Inserting movies into PostgreSQL...")
    movies_df.to_sql("movies", con=engine, if_exists="append", index=False, chunksize=1000)

    print("Inserting ratings into PostgreSQL (100,000 rows)...")
    ratings_df.to_sql("ratings", con=engine, if_exists="append", index=False, chunksize=5000)

    print("\nAll data successfully ingested into PostgreSQL!")


if __name__ == "__main__":
    download_and_extract()
    users_df, movies_df, ratings_df = parse_and_clean_data()
    ingest_to_postgres(users_df, movies_df, ratings_df)
