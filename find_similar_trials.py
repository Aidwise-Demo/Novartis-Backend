import pandas as pd
from similarity_calculator import calculate_similarity
from db_data_retriever import load_table_from_db


def find_top_similar_trials(input_df, disease):
    """
    Finds the top similar trials based on the input data and calculates similarity
    scores with database records for the specified disease.

    Args:
        input_df (pd.DataFrame): DataFrame containing the input trial data.
        disease (str): The disease for which the trial data is processed.

    Returns:
        pd.DataFrame: A DataFrame containing the original data along with
                      calculated similarity scores for each trial.
    """

    # Step 1: Load the database table for embedding data based on disease
    table_name = "embedding"
    db_data = load_table_from_db(table_name, params=(disease,))

    # Check if data exists in the database
    if db_data is None or db_data.empty:
        print("No data found in the database for the specified disease.")
        return pd.DataFrame()  # Return an empty DataFrame if no data is found

    # Step 2: Define the columns to calculate similarity for
    columns_to_embed = [
        'Drug', 'Trial_Phase', 'Population_Segment', 'Disease_Category', 'Primary_Phrases',
        'Secondary_Phrases', 'Inclusion_Phrases', 'Exclusion_Phrases', 'IAge', 'IGender',
        'EAge', 'EGender'
    ]

    # Step 3: Calculate cosine similarity between input data and database records
    similarities = calculate_similarity(input_df, db_data, columns_to_embed)

    # Step 4: Add calculated similarity scores to the original database data
    result_df = pd.concat([db_data, pd.DataFrame(similarities)], axis=1)

    return result_df
