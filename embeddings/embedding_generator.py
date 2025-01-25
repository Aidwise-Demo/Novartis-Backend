import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel

# Initialize the tokenizer and model for embedding generation
tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
model = AutoModel.from_pretrained("medicalai/ClinicalBERT")

# Set the model to evaluation mode to disable dropout and other training-specific behaviors
model.eval()

def generate_input_embeddings(input_data, columns_to_embed):
    """
    Generates embeddings for the specified columns of input data using ClinicalBERT.

    This function processes each column in `columns_to_embed` by converting the text data into embeddings
    using the ClinicalBERT model. It returns a DataFrame containing the original input data along with
    the corresponding embeddings for each specified column.

    Args:
        input_data (dict): Dictionary containing input data for embedding generation.
                            The keys should match the column names in `columns_to_embed`.
        columns_to_embed (list): List of column names for which embeddings should be generated.

    Returns:
        pd.DataFrame: A DataFrame containing the original input data along with the generated embeddings
                      for each specified column.
    """
    embeddings = {}  # Dictionary to store embeddings for each column

    for column in columns_to_embed:
        # Retrieve text data for the column, defaulting to 'unknown' if missing
        text = input_data.get(column, "unknown")

        # Ensure the text is in a list format for processing
        text = [text] if isinstance(text, str) else ["unknown"]

        # Tokenize the text and create tensor inputs for the model
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)

        # Generate embeddings without calculating gradients (for inference)
        with torch.no_grad():
            outputs = model(**inputs)

        # Compute the average embedding for the tokenized sequence
        embeddings[f"{column}_embeddings"] = outputs.last_hidden_state.mean(dim=1).numpy()

    # Convert the embeddings dictionary into a DataFrame
    embedding_df = pd.DataFrame({key: [value] for key, value in embeddings.items()})

    # Create a DataFrame for the original input data
    input_df = pd.DataFrame([input_data])

    # Concatenate the original input DataFrame with the embeddings DataFrame
    return pd.concat([input_df, embedding_df], axis=1)
