# Real-Time AI Movie Recommendation Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Re--Ranking-green)](https://lightgbm.readthedocs.io/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20Search-red)](https://qdrant.tech/)
[![Kafka](https://img.shields.io/badge/Apache%20Kafka-Event%20Streaming-black?logo=apachekafka)](https://kafka.apache.org/)
[![Redis](https://img.shields.io/badge/Redis-In--Memory%20Cache-DC382D?logo=redis)](https://redis.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Production%20Serving-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)](https://www.docker.com/)

Architected and deployed a **production-grade, two-stage real-time movie recommendation platform** inspired by Netflix's and YouTube's industrial architectures, achieving **32.96ms p95 end-to-end latency**, a **100.0% request success rate**, and **0.8963 ROC-AUC ranking precision** under concurrent load, by combining PyTorch Two-Tower deep retrieval, Qdrant vector indexing, LightGBM contextual re-ranking, and Kafka-Redis dynamic session adaptation.

---

## 🏗️ Architecture Overview

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

## 🚀 Key Engineering Accomplishments

### 1. Deep Candidate Retrieval & Vector Search (Stage 1)
- **Reduced candidate retrieval latency to < 8ms** across a 64-dimensional semantic embedding space, by training a **PyTorch Two-Tower deep neural network** that maps user watch histories and movie metadata into normalized dense vectors, indexed in **Qdrant Vector DB** using Cosine Approximate Nearest Neighbor (ANN) search.
- **Overcame the classic collaborative filtering cold-start barrier**, by incorporating content features (genre encoding, release decade, demographic signals) directly into tower projection layers, enabling relevant recommendations even for sparse-interaction items.

### 2. Contextual Re-Ranking & Scoring (Stage 2)
- **Achieved 0.8963 ROC-AUC ranking precision** with **< 2ms inference scoring latency**, by engineering a **LightGBM gradient-boosted decision tree ranker** trained on 8 tabular interaction features, including dynamic genre affinities, user-item rating discrepancies, and historical popularity percentiles.
- **Maximized end-to-end throughput without sacrificing recommendation accuracy**, by decoupling candidate generation (retrieving 30 items) from fine-grained scoring—evaluating complex cross-features only on pre-filtered candidates.

### 3. Event-Driven Streaming & Dynamic Session Adaptation
- **Achieved sub-50ms real-time session taste adaptation without model retraining**, by deploying an asynchronous event-streaming pipeline with **Apache Kafka** (KRaft mode) and a background consumer updating an in-memory **Redis** session store upon every watch event.
- **Eliminated database lock contention and write bottlenecks during traffic spikes**, by buffering live interaction events (`POST /events/watch`) through distributed Kafka partitions, isolating high-frequency ingestion from persistent storage.

### 4. High-Throughput Production Serving & Benchmarking
- **Attained 32.96ms p95 latency and 100.0% request success rate** under concurrent load, by building an asynchronous **FastAPI** gateway utilizing non-blocking coroutines, connection pooling, and sub-1ms Redis in-memory session lookups.
- **Validated production SLAs under concurrent stress**, by executing automated load benchmarks (`benchmarks/benchmark.py`) proving a **19.18ms mean server ML processing time** and sub-22ms median response times across randomized user queries.

---

## 📊 Empirical Benchmark Results

Benchmarked using concurrent load generation across 100 randomized user queries:

| Metric | Measured Result | Production Target SLA | Status |
| :--- | :--- | :--- | :--- |
| **Server ML Processing Time** | **19.18 ms** | `< 25 ms` | ✅ Within SLA |
| **End-to-End Average Latency** | **21.54 ms** | `< 30 ms` | ✅ Within SLA |
| **Median (p50) Latency** | **22.17 ms** | `< 25 ms` | ✅ Within SLA |
| **95th Percentile (p95) Latency** | **32.96 ms** | `< 35 ms` | ✅ Within SLA |
| **Successful Response Rate** | **100.0%** | `99.9%` | ✅ Perfect (Zero Errors) |
| **Stage 2 Model ROC-AUC** | **0.8963** | `> 0.80` | ✅ High Discriminative Power |

---

## 🛠️ Tech Stack & Architecture Rationale

| Layer | Technology | Primary Role | Architectural Advantage |
| :--- | :--- | :--- | :--- |
| **Machine Learning** | PyTorch, LightGBM, Scikit-learn | Two-Tower Embeddings & Re-ranking | Deep semantic retrieval paired with microsecond GBDT tabular scoring |
| **Vector Database** | Qdrant | Dense Vector ANN Search | Sub-8ms similarity search over 64-dim embeddings |
| **Relational Storage** | PostgreSQL 16 | Catalog & Interaction History | ACID-compliant transactional storage for MovieLens metadata |
| **In-Memory Store** | Redis 7 | Feature Store & Session Cache | Sub-1ms RAM lookups for real-time user context |
| **Event Streaming** | Apache Kafka (KRaft mode) | Interaction Event Ingestion | Decouples high-volume click/watch streams from backend storage |
| **API Serving** | FastAPI, Uvicorn | REST API Gateway | High-concurrency async ASGI server with automated OpenAPI docs |
| **Containerization** | Docker, Docker Compose | Multi-Service Orchestration | Reproducible local and production deployment in a single command |

---

## ⚡ Quickstart Guide

### 1. Start Infrastructure Containers
```bash
docker compose up -d
```
*Starts PostgreSQL (`5432`), Redis (`6379`), Qdrant (`6333`), and Kafka (`9092`).*

### 2. Install Dependencies
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Ingest Data & Train Pipeline
```bash
# 1. Download MovieLens dataset and ingest into PostgreSQL
python src/ingest_data.py

# 2. Train PyTorch Two-Tower candidate retrieval model
python src/train_two_tower.py

# 3. Generate and index movie embeddings into Qdrant
python src/index_qdrant.py

# 4. Train Stage 2 LightGBM Re-Ranker
python src/train_ranker.py
```

### 4. Launch FastAPI Server
```bash
python -m uvicorn src.api:app --reload
```
Interactive Swagger documentation is available at **[http://localhost:8000/docs](http://localhost:8000/docs)**.

### 5. Run Performance Benchmark
```bash
python benchmarks/benchmark.py
```

---

## 📜 License
This project is licensed under the [MIT License](LICENSE). Created by [Garvit Singh](https://github.com/GarvitSingh13).
