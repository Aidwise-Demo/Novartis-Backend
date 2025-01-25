import pandas as pd
from scoring.score_cleaning import update_similarity_on_unknown, update_unknown_to_na
from scoring.weight_normalization import adjust_weights_based_on_unknown

def similarity_aggregation(df, input_df, NCT_Number = None):
    # Step 1: Drop unnecessary columns containing embeddings
    columns_to_drop = [
        'Drug_embeddings', 'Trial_Phase_embeddings', 'Population_Segment_embeddings',
        'Disease_Category_embeddings', 'Primary_Phrases_embeddings', 'Secondary_Phrases_embeddings',
        'Inclusion_Phrases_embeddings', 'Exclusion_Phrases_embeddings',
        'IAge_embeddings', 'IGender_embeddings', 'EAge_embeddings', 'EGender_embeddings'
    ]
    df.drop(columns=columns_to_drop, inplace=True)  # Drop the embedding columns
    input_df.drop(columns=columns_to_drop, inplace=True)
    input_df.fillna("unknown")

    missing_columns = [col for col in df.columns if col not in input_df.columns]

    # Step 2: Add missing columns to `input_df` with a default value of 1
    for col in missing_columns:
        input_df[col] = 1

    df = pd.concat([df, input_df], ignore_index=True)

    # Step 2: Calculate Inclusion and Exclusion Criteria Similarities
    df['Inclusion_Criteria_similarity'] = (
        0.4 * (df['IAge_similarity'] + df['IGender_similarity']) +
        0.2 * df['Inclusion_Phrases_similarity']
    )
    df['Exclusion_Criteria_similarity'] = (
        0.4 * (df['EAge_similarity'] + df['EGender_similarity']) +
        0.2 * df['Exclusion_Phrases_similarity']
    )

    # Step 3: Calculate Study Title Similarity
    df['Study_Title_similarity'] = (
        0.4 * (df['Drug_similarity'] + df['Disease_Category_similarity']) +
        0.2 * df['Population_Segment_similarity']
    )

    # Assign Primary and Secondary Outcome Measures similarities
    df['Primary_Outcome_Measures_similarity'] = df['Primary_Phrases_similarity']
    df['Secondary_Outcome_Measures_similarity'] = df['Secondary_Phrases_similarity']

    # Step 4: Recalculate Overall Similarity
    # Load weights and normalize
    weights_df = pd.read_excel("scoring/weights.xlsx")  # Replace with the actual file path
    weights_df['Normalized_Weight'] = weights_df['Weight'] / weights_df['Weight'].sum()
    weights_dict = weights_df.set_index('Column_Name')['Normalized_Weight'].to_dict()

    similarity_columns = [
        col for col in weights_dict.keys() if col in df.columns
    ]

    # Calculate the initial overall similarity
    df['Overall_similarity'] = df[similarity_columns].apply(
        lambda row: sum(row[col] * weights_dict[col] for col in similarity_columns if pd.notna(row[col])),
        axis=1
    )
    # Sort DataFrame in descending order based on Overall_similarity
    df = df.sort_values(by="Overall_similarity", ascending=False)

    if NCT_Number == "None":
        # If NCT_Number is None, take the first row of the DataFrame
        first_row_df = df.iloc[0:1].reset_index(drop=True)
        first_row_df = first_row_df.fillna("unknown")
        df = df.iloc[1:].reset_index(drop=True)

    else:
        # Find the row where 'NCT_Number' matches the input
        first_row_df = df[df["NCT_Number"] == NCT_Number]

        if first_row_df.empty:
            print(f"No matching row found for NCT_Number: {NCT_Number}")
        else:
            # Fill missing values in the first row with "unknown"
            first_row_df = first_row_df.replace('', 'unknown').fillna('unknown')

            # Drop the matching row from the DataFrame
            df = df.drop(first_row_df.index).reset_index(drop=True)

    # Update similarity columns for "unknown" handling
    df = update_similarity_on_unknown(df)
    normalized_weights_dict = adjust_weights_based_on_unknown(first_row_df, weights_dict)

    # Recalculate similarity columns based on the updated weights
    similarity_columns = [
        col for col in normalized_weights_dict.keys() if col in df.columns
    ]

    # Convert all similarity columns to numeric and coerce non-numeric values to NaN
    for col in similarity_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')  # Coerce non-numeric to NaN

    # Fill NaN values with 0 (or another default value)
    df[similarity_columns] = df[similarity_columns].fillna(0)

    # Recalculate Overall_similarity
    df['Overall_similarity'] = df[similarity_columns].apply(
        lambda row: sum(
            row[col] * normalized_weights_dict.get(col, 0)  # Multiply each similarity score by its weight
            for col in similarity_columns
        ),
        axis=1
    )

    # Step 6: Select the Top 10 Rows Based on Overall Similarity
    top_10_df = df.nlargest(10, 'Overall_similarity')
    top_10_df = update_unknown_to_na(first_row_df, top_10_df)

    # Drop unnecessary columns from the top 10 DataFrame
    columns_to_drop = [
        'SerialNumber', 'Trial_Phase', 'Population_Segment', 'Disease_Category', 'Primary_Phrases',
        'Secondary_Phrases', 'Inclusion_Phrases', 'Exclusion_Phrases', 'IAge',
        'IGender', 'EAge', 'EGender', 'Trial_Phase_similarity', 'Population_Segment_similarity',
        'Disease_Category_similarity', 'Primary_Phrases_similarity',
        'Secondary_Phrases_similarity', 'Inclusion_Phrases_similarity',
        'Exclusion_Phrases_similarity', 'IAge_similarity', 'IGender_similarity',
        'EAge_similarity', 'EGender_similarity', 'overall_similarity'  # Remove these columns for final output
    ]
    top_10_df.drop(columns=columns_to_drop, inplace=True)


    return top_10_df
