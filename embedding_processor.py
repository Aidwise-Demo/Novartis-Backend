import pandas as pd
from embedding_generator import generate_input_embeddings

def process_and_generate_embeddings(input_data):
    """
    Process the input data and generate embeddings for specified columns.

    This function takes a DataFrame containing input data, converts it to a dictionary, and then generates
    embeddings for specified columns using a pre-trained model. It returns a DataFrame containing the
    original input data along with the corresponding embeddings for each specified column.

    Args:
        input_data (pd.DataFrame): A DataFrame containing input data for embedding generation.

    Returns:
        pd.DataFrame: A DataFrame with input data and corresponding embeddings for specified columns.
    """
    # Define the columns for which embeddings should be generated
    columns_to_embed = [
        'Drug', 'Trial_Phase', 'Population_Segment', 'Disease_Category', 'Primary_Phrases',
        'Secondary_Phrases', 'Inclusion_Phrases', 'Exclusion_Phrases', 'IAge', 'IGender',
        'EAge', 'EGender'
    ]

    # Convert input DataFrame to dictionary format (assuming only one record is provided)
    input_dict = input_data.to_dict(orient='records')[0]

    # Generate embeddings using the helper function
    input_df = generate_input_embeddings(input_dict, columns_to_embed)

    # Return the DataFrame with input data and embeddings
    return input_df
