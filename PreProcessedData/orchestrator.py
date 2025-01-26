import pandas as pd
from pre_processing import Dataset  # Import Dataset class for processing
from condition_disease_mapping import filter_and_split_conditions
from embeddings_processor_and_generator import (
    get_batch_embeddings,
    save_embeddings_to_db,
    create_embeddings_table
)
from phrases_tagging import process_and_tag_keywords
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# File path for the input dataset
file_path = r'D:\Aidwise\Novartis\Main\Code\Raw Files\usecase1.xlsx'

# Step 1: Filter and split conditions
try:
    logger.info("Filtering and splitting conditions from the dataset.")
    result_df = filter_and_split_conditions(file_path)
    logger.info("Conditions filtered and split successfully.")
except Exception as e:
    logger.error(f"Error during condition filtering: {str(e)}")
    raise

# Step 2: Preprocess the dataset
try:
    logger.info("Initializing dataset processing.")
    datasetProcessor = Dataset("Hypertension", result_df)
    resultDF = datasetProcessor.getProcessedDataset()

    if not resultDF.empty:
        logger.info("Dataset processing completed successfully.")
    else:
        logger.warning("Processed dataset is empty.")
except Exception as e:
    logger.error(f"Dataset processing failed: {str(e)}")
    raise

# Step 3: Tag phrases with keywords
try:
    logger.info("Starting keyword tagging for phrases.")
    result_df = process_and_tag_keywords(result_df, "Hypertension")
    logger.info("Keyword tagging completed successfully.")
except Exception as e:
    logger.error(f"Keyword tagging failed: {str(e)}")
    raise

# Step 4: Create embeddings table in the database (if it doesn't already exist)
try:
    logger.info("Creating embeddings table in the database.")
    create_embeddings_table()
    logger.info("Embeddings table created successfully.")
except Exception as e:
    logger.error(f"Failed to create embeddings table: {str(e)}")
    raise

# Step 5: Check if the dataset contains records
if result_df.empty:
    logger.warning("No records found in the dataset. Exiting the workflow.")
else:
    # Columns requiring embeddings
    columns_to_embed = [
        'Drug', 'Trial_Phase', 'Population_Segment', 'Disease_Category',
        'Primary_Phrases', 'Secondary_Phrases', 'Inclusion_Phrases',
        'Exclusion_Phrases', 'IAge', 'IGender', 'EAge', 'EGender'
    ]

    # Step 6: Generate embeddings for specified columns
    try:
        logger.info("Generating embeddings for specified columns.")
        for column in columns_to_embed:
            if column in result_df.columns:
                embeddings = get_batch_embeddings(result_df[column].tolist())
                result_df[f'{column}_embeddings'] = [emb.numpy() for emb in embeddings]
            else:
                logger.warning(f"Column '{column}' not found in the dataset. Skipping embedding generation.")
        logger.info("Embeddings generated successfully.")
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {str(e)}")
        raise

    # Step 7: Save the data with embeddings to the database
    try:
        logger.info("Saving embeddings to the database.")
        save_embeddings_to_db(result_df)
        logger.info("Embeddings saved to the database successfully.")
    except Exception as e:
        logger.error(f"Failed to save embeddings to the database: {str(e)}")
        raise

print("Workflow completed successfully.")
