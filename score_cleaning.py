import pandas as pd


def update_similarity_on_unknown(df):
    """
    This function checks each column in the DataFrame for the value "unknown".
    If "unknown" is found in any column, it sets the corresponding similarity column (column_name_similarity) to 0 for that row.

    Args:
        df (pd.DataFrame): Input DataFrame with columns containing similarity values and other data.

    Returns:
        pd.DataFrame: The modified DataFrame with similarity values updated to 0 where "unknown" is found.
    """

    # Iterate over each column in the DataFrame
    for column in df.columns:
        # Check if any value in the column is "unknown"
        if df[column].eq("unknown").any():
            # Find the corresponding similarity column for the current column
            similarity_column = f"{column}_similarity"
            if similarity_column in df.columns:
                # Set the similarity column to 0 for rows where the value is "unknown"
                df.loc[df[column] == "unknown", similarity_column] = "NA"

    # Return the modified DataFrame
    return df
