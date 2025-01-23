from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import mysql.connector
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()


# Function to establish a MySQL connection
def get_db_connection():
    try:
        return mysql.connector.connect(
            host=os.getenv('host'),
            user=os.getenv('user'),
            password=os.getenv('password'),
            database=os.getenv('database'),
            port=os.getenv('port')
        )
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None


# Function to extract data from the MySQL database
def extract_table_from_db(disease):
    conn = get_db_connection()
    if conn is None:
        print("Failed to connect to the database.")
        return None

    try:
        query = "SELECT * FROM embedding WHERE LOWER(Disease) = %s"
        df = pd.read_sql(query, conn, params=(disease.lower(),))
        return df
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None
    finally:
        conn.close()


def calculate_similarity(input_embeddings, db_embeddings, columns_to_embed):
    similarities = []

    for _, row in db_embeddings.iterrows():
        row_similarities = {}
        overall_similarity = 0
        for column in columns_to_embed:
            input_emb = input_embeddings[f"{column}_embeddings"].values[0]  # Fetch from input_df
            if isinstance(input_emb, np.ndarray):
                input_emb = input_emb.reshape(1, -1)

            # Convert database embedding from string/serialized format
            db_emb = np.frombuffer(row[f"{column}_embeddings"], dtype=np.float32).reshape(1, -1)

            # Calculate cosine similarity
            similarity = cosine_similarity(input_emb, db_emb)[0][0]
            row_similarities[f"{column}_similarity"] = similarity
            overall_similarity += similarity

        row_similarities["overall_similarity"] = overall_similarity / len(columns_to_embed)
        similarities.append(row_similarities)

    return similarities



# Main function to process the input and calculate similarities
def find_top_similar_trials(input_df, disease):
    # Step 1: Extract data from the database
    db_data = extract_table_from_db(disease)
    if db_data is None or db_data.empty:
        print("No data found in the database.")
        return

    # Step 2: Generate embeddings for the input data
    columns_to_embed = [
        'Drug', 'Trial_Phase', 'Population_Segment', 'Disease_Category', 'Primary_Phrases',
        'Secondary_Phrases', 'Inclusion_Phrases', 'Exclusion_Phrases', 'IAge', 'IGender',
        'EAge', 'EGender'
    ]

    # Step 4: Calculate similarities
    similarities = calculate_similarity(input_df, db_data, columns_to_embed)

    # Step 4: Add similarities to the database table
    db_data = pd.concat([db_data, pd.DataFrame(similarities)], axis=1)

    return db_data

