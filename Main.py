import pandas as pd
import math
import time  # For time measurement
from entity_extractor import entity_extraction
from phrases_extractor import tag_dataframe_with_phrases
from embedding_processor import process_and_generate_embeddings
from find_similar_trials import find_top_similar_trials
from score_aggregation import similarity_aggregation
import os
from dotenv import load_dotenv
from fill_na_nan import replace_none_nan_with_na

# Load environment variables from the .env file
load_dotenv()


def trials_extraction(
        NCT_Number=None,
        Study_Title=None,
        Primary_Outcome_Measures=None,
        Secondary_Outcome_Measures=None,
        Inclusion_Criteria=None,
        Exclusion_Criteria=None
):
    """
    Extracts clinical trial data, processes embeddings, and finds top similar trials.

    Parameters:
    - NCT_Number (str or None): Clinical trial identifier.
    - Study_Title (str or None): Title of the study.
    - Primary_Outcome_Measures (str or None): Primary outcome measures.
    - Secondary_Outcome_Measures (str or None): Secondary outcome measures.
    - Inclusion_Criteria (str or None): Inclusion criteria for the trial.
    - Exclusion_Criteria (str or None): Exclusion criteria for the trial.

    Returns:
    - final_similarity (pd.DataFrame): A DataFrame containing aggregated similarity results.

    Raises:
    - ValueError: If none of the study parameters are provided.
    """

    # Validate input: Ensure at least one argument is provided
    if all(arg is None for arg in [
        Study_Title, Primary_Outcome_Measures, Secondary_Outcome_Measures, Inclusion_Criteria, Exclusion_Criteria
    ]):
        raise ValueError("At least one argument must be provided.")

    # Replace None or NaN values with "NA" in input arguments
    NCT_Number = replace_none_nan_with_na(NCT_Number)
    Study_Title = replace_none_nan_with_na(Study_Title)
    Primary_Outcome_Measures = replace_none_nan_with_na(Primary_Outcome_Measures)
    Secondary_Outcome_Measures = replace_none_nan_with_na(Secondary_Outcome_Measures)
    Inclusion_Criteria = replace_none_nan_with_na(Inclusion_Criteria)
    Exclusion_Criteria = replace_none_nan_with_na(Exclusion_Criteria)

    # Step 1: Entity extraction from Study Title using LLM
    extracted_entities = entity_extraction(Study_Title)

    # Replace None or NaN in extracted entities with "NA"
    disease = replace_none_nan_with_na(extracted_entities['Disease'])
    disease_category = replace_none_nan_with_na(extracted_entities['Disease_Category'])
    drug = replace_none_nan_with_na(extracted_entities['Study_Title_Entities']['Drug'])
    trial_phase = replace_none_nan_with_na(extracted_entities['Study_Title_Entities']['Trial Phase'])
    population_segment = replace_none_nan_with_na(extracted_entities['Study_Title_Entities']['Population Segment'])

    if disease in [None, "NA", "nan"]:
        return "The model is trained on Ulcerative Colitis, Hypertension, and Alzheimer. Please provide relevant data for these diseases."

    # Step 2: Create a DataFrame with trial information
    data = {
        'NCT_Number': [NCT_Number],
        'Study_Title': [Study_Title],
        'Primary_Outcome_Measures': [Primary_Outcome_Measures],
        'Secondary_Outcome_Measures': [Secondary_Outcome_Measures],
        'Inclusion_Criteria': [Inclusion_Criteria],
        'Exclusion_Criteria': [Exclusion_Criteria],
        'Disease': [disease],
        'Disease_Category': [disease_category],
        'Drug': [drug],
        'Trial_Phase': [trial_phase],
        'Population_Segment': [population_segment]
    }
    df = pd.DataFrame(data)  # Convert dictionary to a DataFrame

    # Step 3: Tag the DataFrame with relevant phrases
    tagged_df = tag_dataframe_with_phrases(df, disease)

    # Step 4: Generate embeddings for the tagged DataFrame
    embeddings_df = process_and_generate_embeddings(tagged_df)

    # Step 5: Find top similar trials based on embeddings
    similarity_df = find_top_similar_trials(embeddings_df, disease)

    # Step 6: Aggregate similarity results for better interpretability
    final_similarity = similarity_aggregation(similarity_df)

    # Return the final similarity DataFrame
    return final_similarity
