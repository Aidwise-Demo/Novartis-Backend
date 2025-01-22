import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
model = AutoModel.from_pretrained("medicalai/ClinicalBERT")

# Set the model to evaluation mode
model.eval()

def get_batch_embeddings(text_list, batch_size=16):
    if not text_list:  # Check if the text_list is empty
        return torch.tensor([])  # Return an empty tensor

    embeddings = []
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors='pt', truncation=True, padding=True, max_length=512)

        with torch.no_grad():
            outputs = model(**inputs)

        # Mean pooling across tokens to get the embeddings
        batch_embeddings = outputs.last_hidden_state.mean(dim=1)
        embeddings.append(batch_embeddings)

    return torch.cat(embeddings, dim=0) if embeddings else torch.tensor([])


def process_and_generate_embeddings(df):

    # List of columns to embed
    columns_to_embed = [
        "Drug", "Trial_Phase", "Population_Segment", "Disease_Category", "Primary_Phrases",
        "Secondary_Phrases", "Inclusion_Phrases", "Exclusion_Phrases",
        "IAge", "IGender", "EAge", "EGender"
    ]

    # Create new columns for embeddings
    for column in columns_to_embed:
        df[column + "_embeddings"] = None  # Initialize the embedding columns as None

    # Process each row individually
    for idx, row in df.iterrows():
        # Process each text column for the current row
        for column in columns_to_embed:
            text = row[column]  # Get the text for the current column
            if isinstance(text, str) and text:  # Ensure the text is a valid string
                embeddings = get_batch_embeddings([text])

                # Store the embeddings in the new embedding column
                embedding_column = column + "_embeddings"
                df.at[idx, embedding_column] = embeddings.numpy().tolist() if embeddings.numel() > 0 else None

    return df

# # Load the Excel file
# file_path = "AZ.xlsx"
# df = pd.read_excel(file_path)
#
# # Process and generate embeddings for the file
# process_and_generate_embeddings("Alzheimer", df)
