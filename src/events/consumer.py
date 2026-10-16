import os
import json
import redis
from kafka import KafkaConsumer

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TOPIC_NAME = "user-interactions"

_redis_client = None


def get_redis_client():
    """Initializes and returns a singleton Redis client."""
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    return _redis_client


def update_user_session_in_redis(user_id: int, movie_id: int):
    """
    Pushes the newly watched movie_id to the user's recent watches list in Redis RAM.
    Keeps the last 10 movies, sets expiration to 24 hours (86,400s).
    """
    r = get_redis_client()
    key = f"user:{user_id}:recent_watches"
    r.lpush(key, int(movie_id))
    r.ltrim(key, 0, 9) # Keep most recent 10 items
    r.expire(key, 86400)


def get_user_recent_watches(user_id: int) -> list:
    """Fetches the user's active session history from Redis RAM in <1ms."""
    r = get_redis_client()
    key = f"user:{user_id}:recent_watches"
    movie_ids = r.lrange(key, 0, -1)
    return [int(mid) for mid in movie_ids]


def clear_user_recent_watches(user_id: int):
    """Clears a user's recent watch session (useful for testing)."""
    r = get_redis_client()
    r.delete(f"user:{user_id}:recent_watches")


def process_kafka_events_once(max_records: int = 10, timeout_ms: int = 2000):
    """
    Consumes pending events from Kafka and writes them to Redis.
    Useful for batch ingestion or polling in workers.
    """
    consumer = KafkaConsumer(
        TOPIC_NAME,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        group_id="recsys-redis-updater",
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        consumer_timeout_ms=timeout_ms
    )
    
    count = 0
    for message in consumer:
        event = message.value
        uid = event.get("user_id")
        mid = event.get("movie_id")
        if uid and mid:
            update_user_session_in_redis(uid, mid)
            count += 1
        if count >= max_records:
            break

    consumer.close()
    return count
