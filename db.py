import psycopg2
import os

def get_connection():
    conn = psycopg2.connect(
        dbname=os.getenv("PG_DB", "ChoSV"),
        user=os.getenv("PG_USER", "postgres"),
        password=os.getenv("PG_PASS", "1234"),
        host=os.getenv("PG_HOST", "localhost"),
        port=os.getenv("PG_PORT", 5432),
    )
    return conn