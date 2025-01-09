import os
import json
import sqlite3

def ensure_directory_structure():
    if not os.path.exists("models"):
        os.makedirs("models")

    if not os.path.exists("credentials.json"):
        with open("credentials.json", "w") as f:
            json.dump({"users": {}}, f)

def ensure_database():
    conn = sqlite3.connect("complaints.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS complaints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            complaint TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()
