import psycopg2
import os
from dotenv import load_dotenv
import json

# Загрузить переменные из .env
load_dotenv()

host=os.getenv("PG_HOST")
port=os.getenv("PG_PORT")
dbname=os.getenv("PG_DB_NAME")
user=os.getenv("PG_USER")
password=os.getenv("PG_PASSWORD")

def get_connection():
    conn = psycopg2.connect(
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=password,
        sslmode="require"  # или verify-full при наличии сертификата
    )
    return conn
def close_connection(conn):
    conn.close()

def get_all_aloys_json():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM public.alloys;")

    cols = [desc[0] for desc in cur.description]
    rows = cur.fetchall()

    alloys = [dict(zip(cols, row)) for row in rows]

    cur.close()
    close_connection(conn)

    return alloys


def get_all_fuel_cells_json():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM public.fuel_cells;")
    
    cols = [desc[0] for desc in cur.description]
    rows = cur.fetchall()

    fuel_cells = [dict(zip(cols, row)) for row in rows]
    cur.close()
    close_connection(conn)

    return fuel_cells