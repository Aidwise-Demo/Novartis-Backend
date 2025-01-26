import os
from dotenv import load_dotenv
import mysql.connector
import pandas as pd

# Load .env file to read environment variables
load_dotenv()

# Function to establish a MySQL connection
def get_db_connection():
    # Establish and return the connection using environment variables
    return mysql.connector.connect(
        host=os.getenv('host'),
        user=os.getenv('user'),
        password=os.getenv('password'),
        database=os.getenv('database'),
        port=os.getenv('port')
    )

# Function to create a new database if it does not exist
def create_database(conn, db_name):
    cursor = conn.cursor()
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_name}")
    cursor.close()

# Function to create a table and insert data from DataFrame
def create_table_and_insert_data(conn, table_name, df):
    cursor = conn.cursor()

    # Ensure 'serial_number' exists in the DataFrame and is used as PRIMARY KEY
    if 'serial_number' not in df.columns:
        raise ValueError("DataFrame must contain 'serial_number' column.")

    # Define columns excluding serial_number
    columns = [col for col in df.columns if col != 'serial_number']
    columns_str = ", ".join(columns)

    # Check if the table already exists
    cursor.execute(f"SHOW TABLES LIKE '{table_name}'")
    table_exists = cursor.fetchone() is not None

    if not table_exists:
        # If the table doesn't exist, create it with serial_number as the PRIMARY KEY
        create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} (" \
                             f"serial_number INT PRIMARY KEY, {', '.join([col + ' TEXT' for col in columns])})"
        cursor.execute(create_table_query)
    else:
        print(f"Table {table_name} already exists, skipping table creation.")

    # Insert data into the table, excluding serial_number from the row
    for _, row in df.iterrows():
        placeholders = ", ".join(["%s"] * len(columns))  # Only placeholders for the other columns
        insert_query = f"INSERT INTO {table_name} (serial_number, {columns_str}) " \
                       f"VALUES (%s, {placeholders})"
        # Insert serial_number explicitly along with other values
        cursor.execute(insert_query, (row['serial_number'], *row[columns].values))

    conn.commit()
    cursor.close()

# Load data from Excel sheet and insert into MySQL
file_path = r'D:\Aidwise\Novartis\Main\Code\DB\Embeddings.xlsx'
table = "embeddings"
df = pd.read_excel(file_path)

# Ensure no missing values by filling them with empty string
df = df.fillna('')

# Ensure that the 'serial_number' column is properly set as an integer
df['serial_number'] = df['serial_number'].astype(int)

# Establish the database connection
conn = get_db_connection()

# Create the database if it doesn't exist (use the name from the .env file)
create_database(conn, os.getenv('database'))

# Now that the database exists, connect to it
conn = get_db_connection()

# Create table and insert data from the DataFrame
create_table_and_insert_data(conn, table, df)

# Close the connection after finishing the task
conn.close()
