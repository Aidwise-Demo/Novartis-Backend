import pandas as pd


# Function to extract Age from tags
def extract_age(tags):
    """
    Extracts age information from a string of tags.
    If a tag contains the keyword 'year', it is returned as the age.

    Args:
        tags (str): A comma-separated string of tags.

    Returns:
        str or None: The extracted age-related tag or None if no match is found.
    """
    if isinstance(tags, str):
        for tag in tags.split(","):
            tag = tag.strip().lower()  # Clean and standardize the tag
            if "year" in tag:  # Check if 'year' is present in the tag
                return tag
    return None  # Return None if no age-related tag is found


# Function to extract Gender from tags
def extract_gender(tags):
    """
    Extracts gender information from a string of tags.
    Identifies if the tags specify 'Male', 'Female', or both.

    Args:
        tags (str): A string of tags.

    Returns:
        str or None: 'Male', 'Female', 'Male & Female', or None if no match is found.
    """
    if isinstance(tags, str):
        tags_lower = tags.lower()  # Standardize case for comparison
        if "male" in tags_lower and "female" in tags_lower:
            return "Male & Female"  # Return if both genders are mentioned
        elif "male" in tags_lower and "female" not in tags_lower:
            return "Male" # Return if only male is mentioned
        elif "female" in tags_lower:
            return "Female"  # Return if only female is mentioned
    return None  # Return None if no gender-related tag is found

# Function to tag the DataFrame with extracted information
def tag_age_gender(df):
    """
    Tags a DataFrame with extracted information from inclusion and exclusion phrases.
    Adds columns for extracted age and gender from inclusion and exclusion phrases.

    Args:
        df (pd.DataFrame): The input DataFrame containing 'Inclusion_Phrases' and 'Exclusion_Phrases'.
        disease_name (str): The name of the disease (not used directly but can be used in future enhancements).

    Returns:
        pd.DataFrame: The DataFrame with additional columns for age and gender information.
    """
    # Extract age and gender from inclusion phrases
    df['IAge'] = df['Inclusion_Phrases'].apply(extract_age)  # Inclusion age
    df['IGender'] = df['Inclusion_Phrases'].apply(extract_gender)  # Inclusion gender

    # Extract age and gender from exclusion phrases
    df['EAge'] = df['Exclusion_Phrases'].apply(extract_age)  # Exclusion age
    df['EGender'] = df['Exclusion_Phrases'].apply(extract_gender)  # Exclusion gender

    return df
