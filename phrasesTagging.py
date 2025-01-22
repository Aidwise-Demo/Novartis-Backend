import pandas as pd
from rapidfuzz import fuzz
import os
from dotenv import load_dotenv
import mysql.connector

# Load .env file
load_dotenv()

# Function to establish a MySQL connection
def get_db_connection():
    return mysql.connector.connect(
        host=os.getenv('host'),
        user=os.getenv('user'),
        password=os.getenv('password'),
        database=os.getenv('database'),
        port=os.getenv('port')
    )

# Function to load keywords from MySQL database filtered by disease
def load_keywords_from_db(table_name, disease):
    conn = get_db_connection()
    query = f"SELECT Keywords FROM {table_name} WHERE LOWER(Disease) = %s"
    df_keywords = pd.read_sql(query, conn, params=(disease.lower(),))
    conn.close()
    return [f'"{word.strip().lower()}"' for word in df_keywords['Keywords'].str.split('|').explode()]

# Function for fuzzy matching
def tag_keywords_rapidfuzz(outcome, keywords, threshold=90):
    matched_keywords = []
    for keyword in keywords:
        keyword = keyword.strip('"')  # Remove quotes for matching
        if fuzz.partial_ratio(outcome, keyword) >= threshold:
            matched_keywords.append(keyword)
    return ", ".join(set(matched_keywords))  # Ensure unique matches

# Function to extract Age
def extract_age(tags):
    if isinstance(tags, str):
        for tag in tags.split(","):
            tag = tag.strip().lower()
            if "year" in tag:
                return tag
    return None


# Function to extract Gender
def extract_gender(tags):
    if isinstance(tags, str):
        tags_lower = tags.lower()
        if "male" in tags_lower and "female" in tags_lower:
            return "Male & Female"
        elif "male" in tags_lower:
            return "Male"
        elif "female" in tags_lower:
            return "Female"
    return None

# Main function to process DataFrame and tag keywords
def tag_dataframe_with_keywords(df, disease_name):
    # Load keywords from database
    primary_secondary_keywords = load_keywords_from_db('outcome_keywords', disease_name)
    inclusion_keywords = load_keywords_from_db('inclusion_keywords', disease_name)
    exclusion_keywords = load_keywords_from_db('exclusion_keywords', disease_name)

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

def tag_dataframe_with_phrases(df, disease_name):
    df = tag_dataframe_with_keywords(df, disease_name)
    # Extract Age and Gender from Inclusion and Exclusion Phrases
    df['IAge'] = df['Inclusion_Phrases'].apply(extract_age)
    df['IGender'] = df['Inclusion_Phrases'].apply(extract_gender)
    df['EAge'] = df['Exclusion_Phrases'].apply(extract_age)
    df['EGender'] = df['Exclusion_Phrases'].apply(extract_gender)

    return df
# # Example usage
# # Assuming df is your input DataFrame and disease_name is provided
# # df = pd.read_excel("path_to_your_excel_file.xlsx", sheet_name="your_sheet_name")
# disease_name = "Ulcerative Colitis"
# # Call the function
# df = tag_dataframe_with_keywords(df, disease_name)
#
# # Output is the modified DataFrame with new columns
# print(df.head())
