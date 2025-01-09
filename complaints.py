import sqlite3
import streamlit as st

# Ensure the database and table exist
def ensure_database():
    conn = sqlite3.connect('complaints.db')
    c = conn.cursor()
    

    
    # Create the table with the correct schema
    c.execute('''CREATE TABLE IF NOT EXISTS complaints
                 (body TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

# Function to store a complaint in the database
def store_complaint(body):
    conn = sqlite3.connect('complaints.db')
    c = conn.cursor()
    c.execute("INSERT INTO complaints (body) VALUES (?)", (body,))
    conn.commit()
    conn.close()

# Function to view complaints from the database (for admin)
def view_complaints():
    conn = sqlite3.connect('complaints.db')
    c = conn.cursor()
    c.execute("SELECT * FROM complaints")
    complaints = c.fetchall()
    conn.close()

    if complaints:
        for complaint in complaints:
            st.write(f"**Complaint**: {complaint[0]}")  # complaint[0] corresponds to the 'body'
            st.write(f"**Timestamp**: {complaint[1]}")
            st.write("---")
    else:
        st.write("No complaints available.")
