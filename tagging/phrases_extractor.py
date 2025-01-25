import pandas as pd
from extraction.metadata_extraction import tag_age_gender  # Function for tagging additional metadata
from tagging.phrases_tagging import tag_phrases  # Function for tagging individual phrases
from database.db_data_retriever import load_table_from_db  # Function for loading data from a database


def process_keywords(keywords_series):
    """
    Process the 'Keywords' series to extract individual keywords,
    strip any leading/trailing spaces, convert them to lowercase,
    and wrap each keyword in double quotes.

    Args:
        keywords_series (pd.Series): A Pandas Series containing keyword strings, separated by '|'.

    Returns:
        list: A list of formatted keywords.
    """
    # Split the keywords by '|' and strip spaces and convert to lowercase
    return [
        f'"{word.strip().lower()}"'
        for word in keywords_series.str.split('|').explode()  # Split by '|' and flatten the list
    ]


# Main function to process DataFrame and tag keywords
def tag_dataframe_with_phrases(df, disease_name):
    """
    Process the provided DataFrame and tag keywords for Primary Outcome Measures,
    Secondary Outcome Measures, Inclusion Criteria, and Exclusion Criteria.

    Args:
        df (pd.DataFrame): The DataFrame containing trial information.
        disease_name (str): The disease name to fetch relevant keywords from the database.

    Returns:
        pd.DataFrame: The DataFrame with tagged keywords for the relevant columns.
    """
    # Load & process relevant keywords from the database for Primary, Secondary, Inclusion, and Exclusion
    primary_secondary_keywords = load_table_from_db("outcome_keywords", params=(disease_name,))
    primary_secondary_keywords = process_keywords(primary_secondary_keywords['Keywords'])

    inclusion_keywords = load_table_from_db('inclusion_keywords', params=(disease_name,))
    inclusion_keywords = process_keywords(inclusion_keywords['Keywords'])

    exclusion_keywords = load_table_from_db('exclusion_keywords', params=(disease_name,))
    exclusion_keywords = process_keywords(exclusion_keywords['Keywords'])

    # Ensure the DataFrame contains the necessary columns for tagging
    required_columns = [
        'Primary_Outcome_Measures', 'Secondary_Outcome_Measures',
        'Inclusion_Criteria', 'Exclusion_Criteria'
    ]

    # Check if all required columns are present in the DataFrame
    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"The DataFrame must have a '{column}' column.")

    # Apply the 'tag_phrases' function to tag phrases in the relevant columns
    # Tagging Primary and Secondary Outcome Measures
    df['Primary_Phrases'] = df['Primary_Outcome_Measures'].fillna('').apply(
        lambda x: tag_phrases(x.lower(), primary_secondary_keywords)  # Convert to lowercase for consistent matching
    )
    df['Secondary_Phrases'] = df['Secondary_Outcome_Measures'].fillna('').apply(
        lambda x: tag_phrases(x.lower(), primary_secondary_keywords)
    )

    # Tagging Inclusion and Exclusion Criteria
    df['Inclusion_Phrases'] = df['Inclusion_Criteria'].fillna('').apply(
        lambda x: tag_phrases(x.lower(), inclusion_keywords)
    )
    df['Exclusion_Phrases'] = df['Exclusion_Criteria'].fillna('').apply(
        lambda x: tag_phrases(x.lower(), exclusion_keywords)
    )

    # Additional metadata tagging (if applicable)
    df = tag_age_gender(df)  # Assuming this adds additional metadata

    # Return the updated DataFrame with tagged phrases
    return df
