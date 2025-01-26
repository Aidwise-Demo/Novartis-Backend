import pandas as pd
from rapidfuzz import fuzz

def process_and_tag_keywords(df, disease_name):
    """
    Process a DataFrame by tagging keywords for specific columns based on disease name.

    Args:
        df (pd.DataFrame): The input DataFrame containing the required columns.
        disease_name (str): The name of the disease to filter keywords.

    Returns:
        pd.DataFrame: The updated DataFrame with tagged keyword columns added.
    """
    # File paths for keyword sources
    primary_secondary_keywords_file = "DB/Outcome_Keywords.xlsx"
    inclusion_criteria_keywords_file = "DB/Inclusion_Keywords.xlsx"
    exclusion_criteria_keywords_file = "DB/Exclsuion_Keywords.xlsx"

    # Function to load keywords from a file filtered by disease
    def load_keywords(file_path, disease):
        df_keywords = pd.read_excel(file_path)
        df_filtered = df_keywords[df_keywords['Disease'].str.lower() == disease.lower()]
        return [f'"{word.strip().lower()}"' for word in df_filtered['Keywords'].str.split('|').explode()]

    # Function for fuzzy matching
    def tag_keywords_rapidfuzz(outcome, keywords, threshold=90):
        matched_keywords = []
        for keyword in keywords:
            keyword = keyword.strip('"')  # Remove quotes for matching
            if fuzz.partial_ratio(outcome, keyword) >= threshold:
                matched_keywords.append(keyword)
        return ", ".join(set(matched_keywords))  # Ensure unique matches

    # Load keywords filtered by disease
    primary_secondary_keywords = load_keywords(primary_secondary_keywords_file, disease_name)
    inclusion_keywords = load_keywords(inclusion_criteria_keywords_file, disease_name)
    exclusion_keywords = load_keywords(exclusion_criteria_keywords_file, disease_name)

    # Ensure required columns are present
    required_columns = [
        'Primary_Outcome_Measures', 'Secondary_Outcome_Measures',
        'Inclusion_Criteria', 'Exclusion_Criteria'
    ]
    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"The DataFrame must have a '{column}' column.")

    # Tagging keywords for Primary and Secondary Outcome Measures
    df['Primary_Phrases'] = df['Primary_Outcome_Measures'].fillna('').apply(
        lambda x: tag_keywords_rapidfuzz(x.lower(), primary_secondary_keywords)
    )
    df['Secondary_Phrases'] = df['Secondary_Outcome_Measures'].fillna('').apply(
        lambda x: tag_keywords_rapidfuzz(x.lower(), primary_secondary_keywords)
    )

    # Tagging keywords for Inclusion and Exclusion Criteria
    df['Inclusion_Phrases'] = df['Inclusion_Criteria'].fillna('').apply(
        lambda x: tag_keywords_rapidfuzz(x.lower(), inclusion_keywords)
    )
    df['Exclusion_Phrases'] = df['Exclusion_Criteria'].fillna('').apply(
        lambda x: tag_keywords_rapidfuzz(x.lower(), exclusion_keywords)
    )

    return df

