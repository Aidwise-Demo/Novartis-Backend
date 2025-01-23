import pandas as pd

def similarity_aggregation(df):
    # Step 1: Drop embeddings columns
    columns_to_drop = [
        'Drug_embeddings', 'Trial_Phase_embeddings', 'Population_Segment_embeddings',
        'Disease_Category_embeddings', 'Primary_Phrases_embeddings', 'Secondary_Phrases_embeddings',
        'Inclusion_Phrases_embeddings', 'Exclusion_Phrases_embeddings',
        'IAge_embeddings', 'IGender_embeddings', 'EAge_embeddings', 'EGender_embeddings'
    ]
    df.drop(columns=columns_to_drop, inplace=True)

    # Step 2: Calculate Inclusion_Criteria_similarity
    df['Inclusion_Criteria_similarity'] = (
        0.4 * (df['IAge_similarity'] + df['IGender_similarity']) +
        0.2 * df['Inclusion_Phrases_similarity']
    )

    # Step 3: Calculate Exclusion_Criteria_similarity
    df['Exclusion_Criteria_similarity'] = (
        0.4 * (df['EAge_similarity'] + df['EGender_similarity']) +
        0.2 * df['Exclusion_Phrases_similarity']
    )

    df['Study_Title_similarity'] = (
        0.4 * (df['Drug_similarity'] + df['Disease_Category_similarity']) +
        0.2 * df['Population_Segment_similarity']
    )

    df['Primary_Outcome_Measures_similarity'] = df['Primary_Phrases_similarity']

    df['Secondary_Outcome_Measures_similarity'] = df['Secondary_Phrases_similarity']

    # Step 4: Recalculate overall_similarity
    # Load weights from Excel (Replace 'weights.xlsx' with your actual file)
    weights_df = pd.read_excel("weights.xlsx")  # Columns: 'Column_Name', 'Weight'
    weights_df['Normalized_Weight'] = weights_df['Weight'] / weights_df['Weight'].sum()

    # Create a dictionary for quick lookup of weights
    weights_dict = weights_df.set_index('Column_Name')['Normalized_Weight'].to_dict()

    # Recalculate overall_similarity using the normalized weights
    similarity_columns = [
        col for col in weights_dict.keys() if col in df.columns
    ]
    df['Overall_similarity'] = df[similarity_columns].apply(
        lambda row: sum(row[col] * weights_dict[col] for col in similarity_columns),
        axis=1
    )

    # Step 5: Pick top 10 rows based on Overall_similarity
    top_10_df = df.nlargest(10, 'Overall_similarity')
    columns_to_drop = [
       'SerialNumber',  'Trial_Phase', 'Population_Segment', 'Disease_Category', 'Primary_Phrases',
       'Secondary_Phrases', 'Inclusion_Phrases', 'Exclusion_Phrases', 'IAge',
       'IGender', 'EAge', 'EGender', 'Trial_Phase_similarity', 'Population_Segment_similarity',
       'Disease_Category_similarity', 'Primary_Phrases_similarity',
       'Secondary_Phrases_similarity', 'Inclusion_Phrases_similarity',
       'Exclusion_Phrases_similarity', 'IAge_similarity', 'IGender_similarity',
       'EAge_similarity', 'EGender_similarity', 'overall_similarity']

    top_10_df.drop(columns=columns_to_drop, inplace=True)

    return top_10_df

# Example usage
# df = pd.read_csv("data.csv")  # Load your DataFrame
# top_10_results = similarity_aggregation(df)
# print(top_10_results)
