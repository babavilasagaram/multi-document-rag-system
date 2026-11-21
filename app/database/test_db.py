from app.database.connection import get_connection

conn = get_connection()

with conn.cursor() as cur:
    cur.execute("SELECT version();")
    print(cur.fetchone())

conn.close()