import pandas as pd


def adjust_weights_based_on_unknown(first_row_df, weights_dict):
    """
    This function adjusts the weights of a dictionary based on the values in the first row of a DataFrame.
    If any column (like "Drug") in the first row contains the value "unknown", the corresponding
    column (like "Drug_similarity") will be removed from the weights dictionary.
    The remaining columns will have their weights normalized such that the sum of the weights equals 1.

    Parameters:
    first_row_df (pd.DataFrame): A DataFrame with the first row containing column values to be checked for "unknown".
    weights_dict (dict): A dictionary where keys are column names (ending with '_similarity') and values are the corresponding weights.

    Returns:
    dict: A new dictionary with the adjusted and normalized weights for the remaining columns.
    """

    # Step 1: Identify columns with "unknown" values in the first_row_df
    # We will check for the first row and identify the base column names that have "unknown"
    unknown_columns = [
        col for col in first_row_df.columns if first_row_df.iloc[0][col] == "unknown"
    ]

    # Step 2: Create a list of columns to remove from the weights_dict by appending "_similarity"
    columns_to_remove = [f"{col}_similarity" for col in unknown_columns]

    # Step 3: Remove these identified columns from the weights_dict
    filtered_weights_dict = {
        col: weight for col, weight in weights_dict.items() if col not in columns_to_remove
    }

    # Step 4: Normalize the weights of the remaining columns
    # We sum the weights of the remaining columns and then normalize them.
    total_weight = sum(filtered_weights_dict.values())

    # If no columns remain after filtering, return an empty dictionary (avoid division by zero)
    if total_weight == 0:
        return {}

    # Normalize each weight by dividing by the total weight to ensure the sum is 1.
    normalized_weights_dict = {
        col: weight / total_weight for col, weight in filtered_weights_dict.items()
    }

    # Return the normalized weights dictionary
    return normalized_weights_dict
