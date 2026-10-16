# Real-Time AI Movie Recommendation Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Re--Ranking-green)](https://lightgbm.readthedocs.io/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20Search-red)](https://qdrant.tech/)
[![Kafka](https://img.shields.io/badge/Apache%20Kafka-Event%20Streaming-black?logo=apachekafka)](https://kafka.apache.org/)
[![Redis](https://img.shields.io/badge/Redis-In--Memory%20Cache-DC382D?logo=redis)](https://redis.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Production%20Serving-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)](https://www.docker.com/)

A production-grade, two-stage real-time movie recommendation platform inspired by Netflix's and YouTube's industrial architecture. Features deep learning candidate retrieval, decision-tree contextual re-ranking, and event-driven user session adaptation with sub-35ms p95 latency.

---

## Architecture Overview

```
                                OFFLINE / TRAINING PIPELINE
                                ───────────────────────────
                     PostgreSQL (MovieLens 100k)
                                 │
                                 ▼
                     PyTorch Two-Tower Network
                    (User Tower + Movie Tower)
                                 │
                                 ▼
                     Generate Movie Embeddings
                                 │
                                 ▼
                     Qdrant Vector Database
                  (Indexed with Cosine Distance)


                                ONLINE SERVING PIPELINE
                                ───────────────────────
   Client Request ──► FastAPI Gateway (/recommend/{user_id})
                            │
                            ├──► 1. Read Redis RAM for active session (<1ms)
                            │
                            ├──► 2. Stage 1: Retrieve 30 candidates from Qdrant (<8ms)
                            │
                            ├──► 3. Stage 2: LightGBM Re-Ranker scores candidates (<2ms)
                            │
                            ▼
                  Return Top 10 Personalized Movies (p95 latency: 32.9ms)


                             REAL-TIME EVENT FEEDBACK LOOP
                             ─────────────────────────────
   User Watch Event ──► FastAPI (/events/watch)
                               │
                               ▼
                       Apache Kafka Topic (`user-interactions`)
                               │
                               ▼
                       Consumer Worker ──► Redis In-Memory Session Store
                                           (Immediate dynamic taste adaptation)
```

---

## Key Features

1. **Two-Stage Recommendation Funnel**:
   * **Stage 1 (Retrieval)**: A **PyTorch Two-Tower Neural Network** maps users and movies into a shared 64-dimensional space. **Qdrant Vector DB** retrieves the top candidates in `<8ms` using Approximate Nearest Neighbors (ANN).
   * **Stage 2 (Re-Ranking)**: A **LightGBM Classifier** evaluates candidates using 8 tabular user, movie, and cross-features (movie popularity, rating discrepancy, genre overlap) to generate calibrated match probabilities (`ROC-AUC: 0.8963`).

2. **Real-Time Session Adaptation**:
   * Uses **Apache Kafka** (KRaft mode) and **Redis** to capture user watch events in real time.
   * Blends instantaneous session intent with historical user profiles, allowing recommendations to adapt in under 50ms without retraining.

3. **High-Throughput Production Serving**:
   * Built on **FastAPI** with async non-blocking endpoints.
   * Interactive Swagger UI available at `http://localhost:8000/docs`.

---

## Empirical Benchmark Results

Benchmarked using concurrent load generation across 100 randomized user queries:

| Metric | Measured Result | Production Target SLA |
| :--- | :--- | :--- |
| **Server ML Processing Time** | **19.18 ms** | `< 25 ms` |
| **End-to-End Average Latency** | **21.54 ms** | `< 30 ms` |
| **Median (p50) Latency** | **22.17 ms** | `< 25 ms` |
| **95th Percentile (p95) Latency** | **32.96 ms** | `< 35 ms` |
| **Successful Response Rate** | **100.0%** | `99.9%` |
| **Stage 2 Model ROC-AUC** | **0.8963** | `> 0.80` |

---

## Tech Stack

* **Machine Learning**: PyTorch, LightGBM, Scikit-learn
* **Databases & Vector Search**: Qdrant Vector DB, PostgreSQL, Redis
* **Streaming & Event Ingestion**: Apache Kafka (KRaft mode)
* **Backend & API Serving**: FastAPI, Uvicorn, SQLAlchemy
* **Infrastructure**: Docker, Docker Compose

---

## Quickstart

### 1. Start Infrastructure Containers
```bash
docker compose up -d
```
Starts PostgreSQL (5432), Redis (6379), Qdrant (6333), and Kafka (9092).

### 2. Install Dependencies
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Ingest Data & Train Models
```bash
# 1. Download MovieLens and ingest into PostgreSQL
python src/ingest_data.py

# 2. Train PyTorch Two-Tower model
python src/train_two_tower.py

# 3. Index movie embeddings into Qdrant
python src/index_qdrant.py

# 4. Train Stage 2 LightGBM Ranker
python src/train_ranker.py
```

### 4. Launch FastAPI Server
```bash
python -m uvicorn src.api:app --reload
```
Open **[http://localhost:8000/docs](http://localhost:8000/docs)** to test live endpoints via Swagger UI.

### 5. Run Performance Benchmark
```bash
python benchmarks/benchmark.py
```
