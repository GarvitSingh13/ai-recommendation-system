import os
from sqlalchemy import create_engine, Column, Integer, String, Float, BigInteger, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker

# Database connection URL (matches docker-compose.yml)
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql+psycopg2://postgres:postgrespassword@localhost:5432/recsys_db"
)

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True, index=True)
    age = Column(Integer, nullable=True)
    gender = Column(String(5), nullable=True)
    occupation = Column(String(50), nullable=True)
    zip_code = Column(String(20), nullable=True)


class Movie(Base):
    __tablename__ = "movies"

    movie_id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    release_date = Column(String(50), nullable=True)
    genres = Column(String(255), nullable=False)


class Rating(Base):
    __tablename__ = "ratings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), index=True, nullable=False)
    movie_id = Column(Integer, ForeignKey("movies.movie_id"), index=True, nullable=False)
    rating = Column(Float, nullable=False)
    timestamp = Column(BigInteger, nullable=False)


def init_db():
    """Creates the tables in PostgreSQL if they do not exist."""
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully!")


def get_db():
    """Yields a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
