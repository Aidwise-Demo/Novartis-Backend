# Import necessary libraries
# First Party Imports
import os
from pathlib import Path

# Third Party Imports
from tqdm import tqdm
import pandas as pd
from pandas.io.parsers import TextFileReader

tqdm.pandas()

# Define file paths
eligibilitiesPath = Path(r"InputData\eligibilities.txt")
usecaseDataPath = Path(r"InputData\usecase_1_.csv")
usecaseDataWithEligibilitiesPath = Path(r"Output\usecase_1_with_eligibilities.csv")
usecaseDataWithEligibilitiesOptimisedPath = Path(r"Output\usecase_1_with_eligibilities_Optimised.csv")
output_dir = Path('Output')
output_dir.mkdir(exist_ok=True)


def split_large_text(row, threshold=100000):
    """
    Splits long text into smaller chunks if it exceeds the threshold length.

    Args:
        row (pd.Series): A row of the DataFrame.
        threshold (int): Maximum allowed length of a cell's content.

    Returns:
        pd.Series: Updated row with split text in separate columns.
    """
    new_columns = {}

    for col in row.index:
        cell = str(row[col])
        if len(cell) > threshold:
            # Split the text into chunks of the threshold length
            num_chunks = len(cell) // threshold + (1 if len(cell) % threshold != 0 else 0)
            new_columns[col] = cell[:threshold]

            # Create new columns for the remaining parts
            for i in range(1, num_chunks):
                new_col_name = f"{col}_part_{i}"
                new_columns[new_col_name] = cell[i * threshold: (i + 1) * threshold]
        else:
            new_columns[col] = cell

    return pd.Series(new_columns)


def optimize_and_save_dataframe(df: pd.DataFrame):
    """
    Optimizes DataFrame by converting object columns to string dtype and saves it as a CSV.

    Args:
        df (pd.DataFrame): The input DataFrame to optimize.
    """
    # Convert object columns to string dtype
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('string')

    # Save the optimized DataFrame to a CSV file
    output_file_path = os.path.join('Output', 'usecase_1_with_eligibilities_Optimised.csv')
    df.to_csv(output_file_path, index=False)


def get_eligibilities(eligibilitiesPath: Path = eligibilitiesPath) -> pd.DataFrame:
    """
    Reads the eligibilities data in chunks and concatenates them into a single DataFrame.

    Args:
        eligibilitiesPath (Path): Path to the eligibilities file.

    Returns:
        pd.DataFrame: Concatenated DataFrame containing all eligibilities data.
    """
    chunk_size = 10_000
    chunks = pd.read_csv(eligibilitiesPath, sep="|", engine='c', chunksize=chunk_size)
    return pd.concat(tqdm(chunks, desc="Reading Eligibilities"))


def get_usecase_data(usecaseDataPath: Path = usecaseDataPath) -> pd.DataFrame:
    """
    Reads the use case data in chunks and drops unnecessary columns.

    Args:
        usecaseDataPath (Path): Path to the use case file.

    Returns:
        pd.DataFrame: DataFrame containing the use case data.
    """
    chunk_size = 1_000
    chunks = pd.read_csv(usecaseDataPath, engine='c', chunksize=chunk_size)
    usecase_data = pd.concat(tqdm(chunks, desc="Reading Usecase Data"))
    return usecase_data.drop(['Unnamed: 0.1', 'Unnamed: 0'], axis=1, errors='ignore')


def get_usecase_data_with_eligibilities(optimized: bool = False) -> pd.DataFrame:
    """
    Reads the use case data with eligibilities in chunks and drops unnecessary columns.

    Args:
        optimized (bool): Whether to read the optimized file.

    Returns:
        pd.DataFrame: DataFrame containing the use case data with eligibilities.
    """
    chunk_size = 10_000
    file_path = usecaseDataWithEligibilitiesOptimisedPath if optimized else usecaseDataWithEligibilitiesPath
    chunks = pd.read_csv(file_path, engine='c', chunksize=chunk_size, dtype=str)

    usecase_with_eligibilities = pd.concat(tqdm(chunks, desc="Reading Usecase (with Eligibilities) Data"))

    # Drop columns not required for further processing
    dropping_columns = [
        'sampling_method', 'gender', 'minimum_age', 'maximum_age', 'healthy_volunteers',
        'population', 'gender_description', 'gender_based', 'adult', 'child', 'older_adult'
    ]
    usecase_with_eligibilities.drop(
        columns=[col for col in dropping_columns if col in usecase_with_eligibilities.columns],
        inplace=True,
        axis=1
    )
    return usecase_with_eligibilities


def merge_usecase_and_eligibilities(usecase_data: pd.DataFrame, eligibilities: pd.DataFrame) -> pd.DataFrame:
    """
    Merges the use case data with eligibilities data on NCT identifiers.

    Args:
        usecase_data (pd.DataFrame): DataFrame containing use case data.
        eligibilities (pd.DataFrame): DataFrame containing eligibilities data.

    Returns:
        pd.DataFrame: Merged DataFrame.
    """
    usecase_with_eligibilities = usecase_data.merge(
        eligibilities, left_on='NCT Number', right_on='nct_id', how='inner'
    )

    # Reorder columns for better readability
    ordered_columns = ['id', 'nct_id', 'NCT Number'] + [
        col for col in usecase_with_eligibilities.columns if col not in ['id', 'nct_id', 'NCT Number']
    ]
    return usecase_with_eligibilities[ordered_columns]


if __name__ == "__main__":
    # Read the use case data with eligibilities
    usecase_data_with_eligibilities = get_usecase_data_with_eligibilities()

    # Keep only necessary columns for processing
    keep_columns = [
        'id', 'nct_id', 'Study Title', 'Study URL', 'Conditions',
        'Study Results', 'Primary Outcome Measures', 'Secondary Outcome Measures', 'criteria'
    ]
    usecase_data_with_eligibilities = usecase_data_with_eligibilities[keep_columns]

    # Apply text splitting for long text fields
    usecase_data_with_eligibilities = usecase_data_with_eligibilities.apply(split_large_text, axis=1)

    # Optimize and save the final DataFrame
    optimize_and_save_dataframe(usecase_data_with_eligibilities)
    print("Processing complete! Optimized DataFrame saved.")
