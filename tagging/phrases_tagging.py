import pandas as pd
from rapidfuzz import fuzz, process  # Importing process for additional rapidfuzz functionalities


# Function for fuzzy matching using RapidFuzz
def tag_phrases(outcome, keywords, threshold=90):
    """
    Perform fuzzy matching to find relevant keywords from a given list.

    Args:
        outcome (str): The text or outcome to match against the list of keywords.
        keywords (list): A list of keywords to compare with the outcome.
        threshold (int): The minimum similarity score (0-100) required for a match. Default is 90.

    Returns:
        str: A comma-separated string of matched keywords. Ensures uniqueness of matched keywords.
    """
    if not isinstance(outcome, str) or not keywords:  # Handle cases where outcome is not a string or keywords is empty
        return ""

    matched_keywords = []  # List to store matched keywords

    # Iterate through the keywords and calculate similarity
    for keyword in keywords:
        keyword = keyword.strip('"')  # Remove unnecessary quotes for accurate matching
        # Use RapidFuzz's `fuzz.partial_ratio` to calculate similarity
        similarity_score = fuzz.partial_ratio(outcome, keyword)

        if similarity_score >= threshold:  # Check if similarity meets or exceeds the threshold
            matched_keywords.append(keyword)

    # Return unique matches as a comma-separated string
    return ", ".join(set(matched_keywords))  # Use set() to ensure uniqueness


# Extended RapidFuzz functionality
def get_best_match(outcome, keywords):
    """
    Find the best matching keyword using RapidFuzz's `process.extractOne`.

    Args:
        outcome (str): The text to match against the list of keywords.
        keywords (list): A list of keywords to compare.

    Returns:
        tuple: A tuple containing the best match and its similarity score, or (None, 0) if no match is found.
    """
    if not isinstance(outcome, str) or not keywords:  # Handle invalid inputs
        return None, 0

    # Use `process.extractOne` to find the best match and its score
    best_match = process.extractOne(outcome, keywords)

    return best_match if best_match else (None, 0)  # Return the match or (None, 0) if no match

