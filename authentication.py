# authentication.py
import json
import bcrypt

CREDENTIALS_FILE = "credentials.json"

def load_credentials():
    with open(CREDENTIALS_FILE, "r") as f:
        return json.load(f)

def save_credentials(credentials):
    with open(CREDENTIALS_FILE, "w") as f:
        json.dump(credentials, f)

def hash_password(password):
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

def verify_password(password, hashed_password):
    return bcrypt.checkpw(password.encode("utf-8"), hashed_password.encode("utf-8"))

def authenticate_user(username, password):
    credentials = load_credentials()
    if username in credentials["users"]:
        return verify_password(password, credentials["users"][username])
    return False

def register_user(username, password):
    credentials = load_credentials()
    if username in credentials["users"]:
        return False  # User already exists
    credentials["users"][username] = hash_password(password)
    save_credentials(credentials)
    return True