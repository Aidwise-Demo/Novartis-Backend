import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def calculate_similarity(input_embeddings, db_embeddings, columns_to_embed):
    """
    Calculates cosine similarity between input embeddings and database embeddings
    for specified columns, and computes an overall similarity score for each row.

    Args:
        input_embeddings (pd.DataFrame): DataFrame containing input embeddings.
        db_embeddings (pd.DataFrame): DataFrame containing embeddings from the database.
        columns_to_embed (list): List of column names for which the similarities are calculated.

    Returns:
        list: A list of dictionaries, each containing similarity scores for individual columns
              and an overall similarity score for each database row.
    """
    similarities = []

    # Iterate over each row in the database to calculate similarities
    for _, row in db_embeddings.iterrows():
        row_similarities = {}  # Initialize a dictionary to store individual column similarities
        overall_similarity = 0  # Initialize the variable to accumulate the overall similarity

        # Iterate over each column for which similarity is calculated
        for column in columns_to_embed:
            # Retrieve the input embedding for the current column
            input_emb = input_embeddings[f"{column}_embeddings"].values[0]
            if isinstance(input_emb, np.ndarray):
                input_emb = input_emb.reshape(1, -1)  # Ensure it is a 2D array for cosine similarity

            # Convert the database embedding from byte format to numpy array
            db_emb = np.frombuffer(row[f"{column}_embeddings"], dtype=np.float32).reshape(1, -1)

            # Compute cosine similarity for the current column's embeddings
            similarity = cosine_similarity(input_emb, db_emb)[0][0]
            row_similarities[f"{column}_similarity"] = similarity  # Store the similarity for the column
            overall_similarity += similarity  # Accumulate the overall similarity

        # Calculate the average overall similarity for this row
        row_similarities["overall_similarity"] = overall_similarity / len(columns_to_embed)
        similarities.append(row_similarities)  # Add the similarities for this row to the list

    return similarities
