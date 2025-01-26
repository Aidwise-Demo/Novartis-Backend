# Standard library imports
import csv
import json
import logging
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from typing import List, Tuple

# Third-party imports
import pandas as pd
import requests
from thefuzz import fuzz
from tqdm import tqdm

# Local imports
from EDA import *

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='\033[1m[%(asctime)s] (%(levelname)s) [%(funcName)s: %(lineno)d]\033[0m - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('disease_dataset.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Dataset:

    """
        The `Dataset` class is responsible for processing and returning a final dataset for a given disease. It handles loading and merging data from various sources, filtering the dataset based on the disease condition, and splitting the inclusion and exclusion criteria. The class provides the following methods:

        - `__init__`: Initializes the `Dataset` object with the given parameters, including the disease, file paths, and output directory.
        - `fuzzySearch`: Performs fuzzy string matching to check if a given text matches the disease search term.
        - `splitCriteria`: Splits the inclusion and exclusion criteria from the input dataframe and renames the columns to a desired format.
        - `separateDataset`: Creates three separate dataframes for outcomes, inclusion, and exclusion criteria.
        - `_loadAndMergeData`: Loads and merges the eligibilities and usecase data.
        - `_filterByDisease`: Filters the dataset by the disease condition.
        - `getProcessedDataset`: Orchestrates the entire dataset processing workflow and returns the final dataset.
    """
    def __init__(self, disease: str, eligibilitiesPath: Path = None, usecaseDataPath: Path = None, outputDir: Path = None):
        """
            Initializes a Dataset object with the given parameters.

            Args:
                disease (str): The disease to focus the dataset on.
                eligibilitiesPath (Path, optional): The path to the eligibilities file. Defaults to Path(r"InputData/eligibilities.txt").
                usecaseDataPath (Path, optional): The path to the usecase data file. Defaults to Path(r"InputData/usecase_1_.csv").
                outputDir (Path, optional): The path to the output directory. Defaults to Path('Output').
        """
        
        self.disease = disease
        self.eligibilitiesPath = eligibilitiesPath if eligibilitiesPath else Path("InputData/eligibilities.txt")
        self.usecaseDataPath = usecaseDataPath if usecaseDataPath else Path("InputData/usecase_1_.csv")
        self.outputDir = outputDir if outputDir else Path('Output')
        self.outputDir.mkdir(exist_ok=True)
        
    def fuzzySearch(self, text: str, searchTerm: str, threshold: int = 80) -> bool:
        """
            Performs fuzzy string matching to check if the given text matches the search term, with an optional threshold. If the text is missing (NaN), it returns False. Otherwise, it returns True if the partial ratio of the text and search term (both converted to lowercase) is greater than or equal to the specified threshold (default is 80).
        """              
        try:
            if pd.isna(text):
                return False
            return fuzz.partial_ratio(text.lower(), searchTerm.lower()) >= threshold
        except Exception as e:
            logger.error(f"Error in fuzzySearch: {str(e)}")
            return False

    def cleanText(self, text: str) -> str:
        """
            Cleans the given text by replacing specific characters and removing multiple spaces. If the input text is missing (NaN), an empty string is returned.

            The function uses a dictionary of replacements to clean the text, including replacing special characters like '≥', '≤', ''', '-', '"', and '~' with their corresponding ASCII equivalents. It also removes any backslash characters.

            After the replacements, the function removes any multiple spaces in the text by joining the split text back together.
        """
        if pd.isna(text):
            return ""
        
        # Dictionary of replacements
        replacements = {
            'â‰¥': '>=',
            'â‰¤': '<=',
            'â€™': "'",
            'â€"': "-",
            'â€"': "-",
            'â€œ': '"',
            'â€': '"',
            '\xa0': ' ',
            '~': ' ',
            '\\': '',
        }
        
        cleaned_text = text
        for old, new in replacements.items():
            cleaned_text = cleaned_text.replace(old, new)
        
        # Remove multiple spaces
        cleaned_text = ' '.join(cleaned_text.split())
        
        return cleaned_text

    def cleanDataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
            Cleans the input DataFrame by applying the `cleanText` function to the specified columns. The function logs the start and completion of the cleaning process, and handles any exceptions that may occur during the cleaning.
            
            Args:
                df (pd.DataFrame): The input DataFrame to be cleaned.
            
            Returns:
                pd.DataFrame: The cleaned DataFrame.
        """

        try:
            logger.info("Starting dataset cleaning process")
            df = df.copy()
            
            columns_to_clean = [
                'Primary_Outcome_Measures',
                'Secondary_Outcome_Measures',
                'Inclusion_Criteria',
                'Exclusion_Criteria'
            ]
            
            for column in columns_to_clean:
                if column in df.columns:
                    logger.info(f"Cleaning column: {column}")
                    df[column] = df[column].apply(self.cleanText)
                    logger.info(f"Successfully cleaned {column}")
                    
            logger.info("Dataset cleaning completed successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error in cleanDataset: {str(e)}")
            raise

    
    def splitCriteria(self, df: pd.DataFrame, keepColumns: List[str]) -> pd.DataFrame:
        """
            Splits the inclusion and exclusion criteria from the input dataframe and renames the columns to a desired format.
        
                Args:
                    df (pd.DataFrame): The input dataframe.
                    keepColumns (List[str]): The list of columns to keep in the output dataframe.
                
                Returns:
                    pd.DataFrame: The dataframe with the inclusion and exclusion criteria split and columns renamed.
        """
        try:
            df = df.copy()
            
            df[['Inclusion_Criteria', 'Exclusion_Criteria']] = df['criteria'].str.split('Exclusion Criteria:~', expand=True, n=1)
            df = df.drop('criteria', axis=1)
            df['Inclusion_Criteria'] = df['Inclusion_Criteria'].str.replace(r'^Inclusion Criteria:~\* ', '', regex=True)
            
            columnMapping = {
                'nct_id': 'NCT_Number',
                'Study Title': 'Study_Title',
                'Primary Outcome Measures': 'Primary_Outcome_Measures',
                'Secondary Outcome Measures': 'Secondary_Outcome_Measures'
            }
            df = df.rename(columns=columnMapping)
            
            keepColumns[:] = [columnMapping.get(col, col) for col in keepColumns]
            keepColumns.extend(['Inclusion_Criteria', 'Exclusion_Criteria'])
            
            return df
            
        except Exception as e:
            logger.error(f"Error in splitCriteria: {str(e)}")
            raise

    @staticmethod
    def separateDataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
            Separates the input dataframe into three separate dataframes: one for the primary and secondary outcome measures, one for the inclusion criteria, and one for the exclusion criteria. The outcome measures dataframe contains the concatenated text of the primary and secondary outcome measures, with any missing values filled with an empty string and the text stripped of leading/trailing whitespace. The inclusion and exclusion criteria dataframes contain the corresponding text from the input dataframe.
        """                
        outcomesDF = pd.DataFrame({
            'NCT_Number': df['NCT_Number'],
            'Text': 'Primary Outcome: ' + df['Primary_Outcome_Measures'].fillna('') + '\nSecondary Outcome: ' + 
                   df['Secondary_Outcome_Measures'].fillna('')
        })
        outcomesDF['Text'] = outcomesDF['Text'].str.strip()
        
        inclusionDF = pd.DataFrame({
            'NCT_Number': df['NCT_Number'],
            'Text': df['Inclusion_Criteria']
        })
        
        exclusionDF = pd.DataFrame({
            'NCT_Number': df['NCT_Number'],
            'Text': df['Exclusion_Criteria']
        })
        
        return outcomesDF, inclusionDF, exclusionDF

    def _loadAndMergeData(self) -> pd.DataFrame:
        """
            Loads and merges the eligibility and usecase data in parallel using ThreadPoolExecutor, then returns the merged dataframe.
        """

        with ThreadPoolExecutor() as executor:
            future_eligibilities = executor.submit(get_eligibilities, eligibilitiesPath=self.eligibilitiesPath)
            future_usecase = executor.submit(get_usecase_data, usecaseDataPath=self.usecaseDataPath)
            
            eligibilities = future_eligibilities.result()
            usecaseData = future_usecase.result()

        return merge_usecase_and_eligibilities(usecaseData, eligibilities)

    def _filterByDisease(self, df: pd.DataFrame) -> pd.DataFrame:
        """
            Filters the input DataFrame `df` to only include rows where the 'Conditions' column matches the `self.disease` attribute using a fuzzy search. If no matching conditions are found, an empty DataFrame is returned.
        """
        conditionMask = df['Conditions'].apply(lambda x: self.fuzzySearch(x, self.disease))
        filteredDF = df[conditionMask]
        
        if filteredDF.empty:
            logger.warning(f"No matching conditions found for disease: {self.disease}")
            return pd.DataFrame()
            
        return filteredDF

    def getProcessedDataset(self) -> pd.DataFrame:
        """
            Processes the dataset for the specified disease, returning a DataFrame containing the relevant data.

            This method first loads and merges the eligibility and usecase data in parallel, then filters the merged data to only include rows where the 'Conditions' column matches the specified disease using a fuzzy search. If no matching conditions are found, an empty DataFrame is returned.

            The final dataset is created by splitting the filtered data into separate DataFrames for the primary and secondary outcome measures, inclusion criteria, and exclusion criteria, and then selecting only the specified columns.

            Raises:
                ValueError: If the disease name is empty.
                Exception: If an error occurs during the dataset processing.

            Returns:
                pd.DataFrame: The processed dataset, or an empty DataFrame if no matching conditions are found.
        """

        try:
            logger.info(f"Starting dataset processing for disease: {self.disease}")
            
            if not self.disease:
                raise ValueError("Disease name cannot be empty.")
            
            keepColumns = [
                'nct_id', 'Study Title', 'Conditions',
                'Primary Outcome Measures', 'Secondary Outcome Measures'
            ]
            
            mergedData = self._loadAndMergeData()
            logger.info(f"Merged data shape: {mergedData.shape}")
            
            filteredData = self._filterByDisease(mergedData)
            if filteredData.empty:
                return pd.DataFrame()
                
            logger.info(f"Records matching disease condition: {len(filteredData)}")
            
            finalDataset = self.splitCriteria(filteredData, keepColumns)[keepColumns]
            finalDataset = self.cleanDataset(finalDataset)
            logger.info(f"Final dataset shape: {finalDataset.shape}")
            logger.info(f"Final columns: {finalDataset.columns.tolist()}")
            
            return finalDataset
            
        except Exception as e:
            logger.error(f"Error in getProcessedDataset: {str(e)}")
            raise
