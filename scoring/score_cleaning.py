import pandas as pd
import numpy as np

def update_similarity_on_unknown(df):
    """
    This function checks each column in the DataFrame for the value "unknown".
    If "unknown" is found in any column, it sets the corresponding similarity column (column_name_similarity) to "NA" for that row.

    Args:
        df (pd.DataFrame): Input DataFrame with columns containing similarity values and other data.

    Returns:
        pd.DataFrame: The modified DataFrame with similarity values updated to "NA" where "unknown" is found.
    """

    # Iterate over each column in the DataFrame
    for column in df.columns:
        # Check if any value in the column is "unknown"
        if df[column].eq("unknown").any():
            # Find the corresponding similarity column for the current column
            similarity_column = f"{column}_similarity"
            if similarity_column in df.columns:
                # Set the similarity column to "NA" for rows where the value is "unknown"
                df.loc[df[column] == "unknown", similarity_column] = "NA"

    # Return the modified DataFrame
    return df


def update_unknown_to_na(df1, df2):
    """
    Updates rows in the second DataFrame (df2) with 'NA' in columns
    where the first row of the first DataFrame (df1) has the value 'unknown', 'Not Available', or 'NA'.
    The column names in df2 are assumed to follow the pattern <column_name>_similarity.

    Parameters:
        df1 (pd.DataFrame): DataFrame with only the first row to check for 'unknown', 'Not Available', or 'NA'.
        df2 (pd.DataFrame): DataFrame to update based on df1.

    Returns:
        pd.DataFrame: Updated df2 with 'NA' in relevant columns.
    """
    # Find columns in df1 where the first row has values 'unknown', 'Not Available', or 'NA'
    unknown_columns = df1.columns[df1.iloc[0].isin(["unknown", "Not Available", "NA"])]

    # Adjust column names for df2 by appending '_similarity'
    similarity_columns = [f"{col}_similarity" for col in unknown_columns if f"{col}_similarity" in df2.columns]

    # Replace corresponding columns in df2 with "NA"
    df2[similarity_columns] = df2[similarity_columns].applymap(lambda x: "NA")

    return df2
