import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

# Initialize tokenizer and model for embedding generation
tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
model = AutoModel.from_pretrained("medicalai/ClinicalBERT")

# Set the model to evaluation mode
model.eval()


# Function to generate and save input embeddings into a DataFrame
def generate_input_embeddings(input_data, columns_to_embed):
    """
    Generate embeddings for the input data and return them in a DataFrame.

    Args:
        input_data (dict): Input dictionary containing data for embedding.
        columns_to_embed (list): List of columns for which embeddings are generated.

    Returns:
        pd.DataFrame: A DataFrame containing input data and corresponding embeddings.
    """
    embeddings = {}
    for column in columns_to_embed:
        text = input_data.get(column, "unknown")  # Replace missing values with 'unknown'
        text = [text] if isinstance(text, str) else ["unknown"]  # Ensure it's a list
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        # Save embedding as numpy array
        embeddings[f"{column}_embeddings"] = outputs.last_hidden_state.mean(dim=1).numpy()

    # Create a DataFrame with original input and embeddings
    embedding_df = pd.DataFrame({key: [value] for key, value in embeddings.items()})
    input_df = pd.DataFrame([input_data])  # Original input as a DataFrame
    return pd.concat([input_df, embedding_df], axis=1)

# Main function to process the input and calculate similarities
def process_and_generate_embeddings(input_data):
    #Generate embeddings for the input data
    columns_to_embed = [
        'Drug', 'Trial_Phase', 'Population_Segment', 'Disease_Category', 'Primary_Phrases',
        'Secondary_Phrases', 'Inclusion_Phrases', 'Exclusion_Phrases', 'IAge', 'IGender',
        'EAge', 'EGender'
    ]
    input_dict = input_data.to_dict(orient='records')[0]
    input_df = generate_input_embeddings(input_dict, columns_to_embed)

    return input_df

