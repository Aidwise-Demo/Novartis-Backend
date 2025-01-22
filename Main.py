import pandas as pd
import math
from entityExtraction import extract_study_title_entities
from phrasesTagging import tag_dataframe_with_phrases
from embeddingGeneration import process_and_generate_embeddings
from similarityCheck import find_top_similar_trials

# Function to replace None and nan with "NA"
def replace_none_nan_with_na(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    return value

def trials_extraction(NCT_Number=None, Study_Title=None, Primary_Outcome_Measures=None, Secondary_Outcome_Measures=None,
                      Inclusion_Criteria=None,
                      Exclusion_Criteria=None):
    if all(arg is None for arg in
           [Study_Title, Primary_Outcome_Measures, Secondary_Outcome_Measures, Inclusion_Criteria, Exclusion_Criteria]):
        raise ValueError("At least one argument must be provided.")

    # Replace None with "NA" in input arguments
    NCT_Number = replace_none_nan_with_na(NCT_Number)
    Study_Title = replace_none_nan_with_na(Study_Title)
    Primary_Outcome_Measures = replace_none_nan_with_na(Primary_Outcome_Measures)
    Secondary_Outcome_Measures = replace_none_nan_with_na(Secondary_Outcome_Measures)
    Inclusion_Criteria = replace_none_nan_with_na(Inclusion_Criteria)
    Exclusion_Criteria = replace_none_nan_with_na(Exclusion_Criteria)

    # Extract entities
    extracted_entities = extract_study_title_entities(Study_Title)
    print(extracted_entities)

    # Replace nan with "NA" for all relevant fields in extracted entities
    disease = replace_none_nan_with_na(extracted_entities['Disease'])
    disease_category = replace_none_nan_with_na(extracted_entities['Disease_Category'])
    drug = replace_none_nan_with_na(extracted_entities['Study_Title_Entities']['Drug'])
    trial_phase = replace_none_nan_with_na(extracted_entities['Study_Title_Entities']['Trial Phase'])
    population_segment = replace_none_nan_with_na(extracted_entities['Study_Title_Entities']['Population Segment'])

    # Create a dictionary for DataFrame
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

    # Convert to DataFrame
    df = pd.DataFrame(data)

    tagged_df = tag_dataframe_with_phrases(df, disease)

    embeddings_df = process_and_generate_embeddings(tagged_df)

    embedding_columns = [
        'Drug_embeddings', 'Trial_Phase_embeddings', 'Population_Segment_embeddings', 'Disease_Category_embeddings',
        'Primary_Phrases_embeddings', 'Secondary_Phrases_embeddings', 'Inclusion_Phrases_embeddings',
        'Exclusion_Phrases_embeddings', 'IAge_embeddings', 'IGender_embeddings', 'EAge_embeddings', 'EGender_embeddings'
    ]

    similarity_df = find_top_similar_trials(embeddings_df, disease, embedding_columns)

    print(similarity_df)

NCT = "1235"
study_title = "Safety and Efficacy of Deep Wave Trabeculoplasty (DWT) in Primary Ocular Hypertension"
primary_outcome = "Percent decrease in IOP and change in dependence on IOP-lowering medications from baseline, 6 months|Intra-procedural and post-procedural adverse events, 6 months"
secondary_outcome = "NA"
inclusion = "Male or female subjects, 18 years or older.~* Subjects diagnosed with either POAG or OHT in both eyes. The diagnosis of POAG must include evidence of:~  1. Optic disc or retinal nerve fiber layer structural abnormalities (substantiated by OCT); and/or~ 2..."
exclusion = "NA"

# df = trials_extraction(NCT, study_title, primary_outcome, secondary_outcome, inclusion, exclusion)



