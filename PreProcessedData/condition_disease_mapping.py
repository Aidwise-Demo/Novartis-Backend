import pandas as pd


def filter_and_split_conditions(file_path):
    """
    Filters and splits the DataFrame based on conditions and their synonyms,
    removes duplicates, and saves the result to an Excel file.

    Parameters:
        file_path (str): The path to the input Excel file.
        output_path (str): The path to save the output Excel file.
        condition_synonyms (dict): A dictionary with disease names as keys and synonyms as values.

    Returns:
        pd.DataFrame: A DataFrame containing the filtered and distinct conditions.
    """
    # Read the input file
    df = pd.read_excel(file_path)

    # Dictionary of conditions and synonyms
    condition_synonyms = {
        "Ulcerative Colitis": [
            "ulcerative colitis", "colitis ulcerosa", "ulcerative", "colitis"
        ],
        "Hypertension": [
            "hypertension", "arterial hypertension", "high blood pressure", "elevated blood pressure",
            "blood pressure high", "hypertensive disorder", "hypertensive disease", "systemic hypertension",
            "HBP", "systemic arterial hypertension", "hypertension arterial", "HTN - hypertension",
            "blood high pressure", "family history of hypertension", "vascular hypertension",
            "hypertensive diseases", "hyperpiesia", "vascular hypertensive disorder", "high blood pressures",
            "hyperpiesis"
        ],
        "Alzheimer": [
            "alzheimer disease", "alzheimer's disease", "alzheimer's dementia", "alzheimers disease",
            "dementia alzheimers", "alzheimer dementia", "dementia of the alzheimer's type", "senile dementia",
            "alzheimer type dementia", "familial alzheimer disease", "dementia alzheimer's type",
            "alzheimer type senile dementia", "alzheimer's diseases", "pN2", "familial alzheimer's disease",
            "dats", "alzheimer syndrome"
        ]
    }

    # Fill NaN values in the 'Conditions' column with an empty string
    df['Conditions'] = df['Conditions'].fillna('')

    # Function to filter rows based on a list of synonyms
    def filter_by_synonyms(df, synonyms, disease_name):
        pattern = '|'.join(synonyms)
        filtered_df = df[df['Conditions'].str.contains(pattern, case=False, regex=True)].copy()
        filtered_df['Disease'] = disease_name
        return filtered_df

    # List to store DataFrames for each disease
    filtered_dfs = []

    # Iterate over the conditions and their synonyms
    for disease, synonyms in condition_synonyms.items():
        filtered_dfs.append(filter_by_synonyms(df, synonyms, disease))

    # Concatenate all filtered DataFrames
    combined_df = pd.concat(filtered_dfs, ignore_index=True)

    # Remove duplicates based on 'Conditions' and 'Disease'
    combined_distinct_df = combined_df.drop_duplicates(subset=['Conditions', 'Disease'])

    return combined_distinct_df

