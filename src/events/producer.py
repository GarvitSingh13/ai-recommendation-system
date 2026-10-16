import os
import json
import time
from kafka import KafkaProducer

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TOPIC_NAME = "user-interactions"

_producer = None


def get_kafka_producer():
    """Initializes and returns a singleton KafkaProducer instance."""
    global _producer
    if _producer is None:
        print(f"Connecting Kafka Producer to {KAFKA_BOOTSTRAP_SERVERS}...")
        _producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            retries=3
        )
    return _producer


def send_user_event(user_id: int, movie_id: int, event_type: str = "watch", rating: float = 5.0) -> dict:
    """
    Publishes an event to Kafka when a user interacts with a movie.
    Example event: {"user_id": 42, "movie_id": 1, "event_type": "watch", "timestamp": 1697500000}
    """
    producer = get_kafka_producer()
    payload = {
        "user_id": int(user_id),
        "movie_id": int(movie_id),
        "event_type": event_type,
        "rating": float(rating),
        "timestamp": time.time()
    }
    
    # Asynchronously produce to Kafka topic
    producer.send(TOPIC_NAME, value=payload)
    producer.flush() # Ensure it's delivered
    return payload
