import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
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


# Function to fetch saved embeddings from MySQL database
def fetch_saved_embeddings(disease):
    conn = get_db_connection()
    if conn is None:
        return pd.DataFrame()  # Return empty DataFrame if connection fails
    try:
        query = "SELECT * FROM embeddings WHERE LOWER(Disease) = %s"
        df_saved = pd.read_sql(query, conn, params=(disease.lower(),))
    finally:
        conn.close()
    return df_saved


# Function to safely evaluate strings into arrays
def safe_eval(val):
    try:
        return np.array(eval(val))
    except Exception as e:
        # print(f"Error evaluating: {val}, error: {e}")
        return np.nan


def compute_similarity(input_embedding, saved_embeddings, input_text=None, comparison_texts=None):
    """
    Calculate cosine similarity with normalization and handle exact matches for 100% similarity.
    """
    if input_embedding is None or saved_embeddings is None or len(saved_embeddings) == 0:
        print("Warning: One or more input embeddings are invalid.")
        return np.zeros(saved_embeddings.shape[0])  # Return a zero vector in case of invalid data

    # Normalize the embeddings
    input_embedding = normalize(input_embedding, axis=1)
    saved_embeddings = normalize(saved_embeddings, axis=1)

    # Compute cosine similarity
    cosine_sim = cosine_similarity(input_embedding, saved_embeddings).flatten()

    if input_text is not None and comparison_texts is not None:
        input_text = str(input_text).strip()
        comparison_texts = [str(text).strip() for text in comparison_texts]
        exact_match_indices = [
            i for i, text in enumerate(comparison_texts) if text.lower() == input_text.lower()
        ]
        for i in exact_match_indices:
            if 0 <= i < len(cosine_sim):  # Ensure the index is valid
                cosine_sim[i] = 1.0  # Assign 100% similarity for exact matches
            else:
                print(f"Warning: Exact match index {i} is out of bounds for cosine_sim of size {len(cosine_sim)}.")

    return cosine_sim


def find_top_similar_trials(input_embeddings, disease, target_columns):
    df_saved = fetch_saved_embeddings(disease)
    if df_saved.empty:
        return pd.DataFrame()  # Return empty DataFrame if no data is retrieved

    similarity_scores = {}
    for column in target_columns:
        if column in input_embeddings.columns and column in df_saved.columns:
            input_emb = input_embeddings[column].values[0]
            if input_emb is None or len(input_emb) == 0:
                print(f"Warning: Input embedding for column {column} is empty or None.")
                continue  # Skip the current column if there's no valid embedding

            saved_emb = np.vstack(df_saved[column].apply(safe_eval).dropna().values)
            if saved_emb.shape[0] == 0:
                print(f"Warning: No valid embeddings found in column {column}.")
                continue  # Skip the current column if there are no valid embeddings

            similarity_scores[column] = compute_similarity(input_emb, saved_emb)

    # Ensure similarity arrays have compatible shapes
    for column in similarity_scores:
        similarity_scores_column = similarity_scores[column]

        # Check if the length of similarity_scores matches the number of rows in df_saved
        if len(similarity_scores_column) == df_saved.shape[0]:
            df_saved[f'{column}_similarity'] = similarity_scores_column
        else:
            print(f"Warning: Length mismatch for {column}. Expected {df_saved.shape[0]} but got {len(similarity_scores_column)}.")
            # Pad with zeros if there's a length mismatch
            similarity_scores_column = np.zeros(df_saved.shape[0])

            # Assign the padded similarity scores to the dataframe
            df_saved[f'{column}_similarity'] = similarity_scores_column

    # Calculate the overall similarity based on weighted contributions from each similarity column
    df_saved['Overall_similarity'] = (
            0.2 * (df_saved.get('Drug_similarity', 0) + df_saved.get('Disease_Category_similarity', 0)) +
            0.15 * (
                    df_saved.get('Population_Segment_similarity', 0) +
                    df_saved.get('Primary_Phrases_similarity', 0) +
                    df_saved.get('Inclusion_Phrases_similarity', 0)
            ) +
            0.1 * df_saved.get('Exclusion_Phrases_similarity', 0) +
            0.05 * df_saved.get('Secondary_Phrases_similarity', 0)
    )

    df_top_10 = df_saved.sort_values(by='Overall_similarity', ascending=False).head(10)
    columns_to_drop = [col for col in df_saved.columns if 'embeddings' in col]
    df_top_10 = df_top_10.drop(columns=columns_to_drop, errors='ignore')

    return df_top_10
