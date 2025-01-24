import os
from dotenv import load_dotenv
import mysql.connector

# Load .env file
load_dotenv()
# Function to establish a MySQL connection
def get_db_connection():
    return mysql.connector.connect(
        host=os.getenv('host'),
        user=os.getenv('user'),
        password=os.getenv('password'),
        database=os.getenv('database'),
        port=os.getenv('port')
    )