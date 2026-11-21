
import os
import psycopg
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not configured.")


def get_connection():
    try:
        conn = psycopg.connect(DATABASE_URL)
        print("Database connected successfully!")
        return conn

    except psycopg.Error:
        print("Database connection failed.")
        raise
