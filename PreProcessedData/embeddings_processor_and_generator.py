import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import mysql.connector
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Initialize tokenizer and model for embedding generation using ClinicalBERT
tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
model = AutoModel.from_pretrained("medicalai/ClinicalBERT")

# Set the model to evaluation mode to prevent training-specific behavior
model.eval()

# Function to establish a MySQL connection
def get_db_connection():
    """
    Establish a connection to the MySQL database using environment variables.
    """
    try:
        return mysql.connector.connect(
            host=os.getenv('host'),
            user=os.getenv('user'),
            password=os.getenv('password'),
            database=os.getenv('database'),
            port=os.getenv('port')
        )
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None

# Function to create the embedding table in the database
def create_embeddings_table():
    """
    Create a MySQL table for storing embeddings if it doesn't already exist.
    """
    conn = get_db_connection()
    if conn is None:
        print("Failed to connect to the database.")
        return
    try:
        cursor = conn.cursor()
        # SQL to create the table with necessary columns and embedding fields
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embedding (
                SerialNumber INT AUTO_INCREMENT PRIMARY KEY,
                NCT_Number TEXT,
                Study_Title TEXT,
                Primary_Outcome_Measures TEXT,
                Secondary_Outcome_Measures TEXT,
                Inclusion_Criteria TEXT,
                Exclusion_Criteria TEXT,
                Disease TEXT,
                Drug TEXT,
                Trial_Phase TEXT,
                Population_Segment TEXT,
                Disease_Category TEXT,
                Primary_Phrases TEXT,
                Secondary_Phrases TEXT,
                Inclusion_Phrases TEXT,
                Exclusion_Phrases TEXT,
                IAge TEXT,
                IGender TEXT,
                EAge TEXT,
                EGender TEXT,
                Drug_embeddings LONGBLOB,
                Trial_Phase_embeddings LONGBLOB,
                Population_Segment_embeddings LONGBLOB,
                Disease_Category_embeddings LONGBLOB,
                Primary_Phrases_embeddings LONGBLOB,
                Secondary_Phrases_embeddings LONGBLOB,
                Inclusion_Phrases_embeddings LONGBLOB,
                Exclusion_Phrases_embeddings LONGBLOB,
                IAge_embeddings LONGBLOB,
                IGender_embeddings LONGBLOB,
                EAge_embeddings LONGBLOB,
                EGender_embeddings LONGBLOB
            );
        """)
        conn.commit()
    except mysql.connector.Error as err:
        print(f"Error: {err}")
    finally:
        conn.close()

# Function to save embeddings to the MySQL database
def save_embeddings_to_db(df):
    """
    Save embeddings and corresponding metadata to the MySQL database.
    """
    conn = get_db_connection()
    if conn is None:
        print("Failed to connect to the database.")
        return

    try:
        cursor = conn.cursor()
        for _, row in df.iterrows():
            # Prepare embeddings for each column as bytes
            embeddings = {col: row[f'{col}_embeddings'].tobytes() for col in [
                'Drug', 'Trial_Phase', 'Population_Segment', 'Disease_Category',
                'Primary_Phrases', 'Secondary_Phrases', 'Inclusion_Phrases',
                'Exclusion_Phrases', 'IAge', 'IGender', 'EAge', 'EGender'
            ]}

            # SQL query to insert the data into the embedding table
            query = """
                INSERT INTO embedding (
                    NCT_Number, Study_Title, Primary_Outcome_Measures, Secondary_Outcome_Measures,
                    Inclusion_Criteria, Exclusion_Criteria, Disease, Drug, Trial_Phase, Population_Segment,
                    Disease_Category, Primary_Phrases, Secondary_Phrases, Inclusion_Phrases, Exclusion_Phrases,
                    IAge, IGender, EAge, EGender, Drug_embeddings, Trial_Phase_embeddings,
                    Population_Segment_embeddings, Disease_Category_embeddings, Primary_Phrases_embeddings,
                    Secondary_Phrases_embeddings, Inclusion_Phrases_embeddings, Exclusion_Phrases_embeddings,
                    IAge_embeddings, IGender_embeddings, EAge_embeddings, EGender_embeddings
                ) VALUES (
                    %(NCT_Number)s, %(Study_Title)s, %(Primary_Outcome_Measures)s, %(Secondary_Outcome_Measures)s,
                    %(Inclusion_Criteria)s, %(Exclusion_Criteria)s, %(Disease)s, %(Drug)s, %(Trial_Phase)s,
                    %(Population_Segment)s, %(Disease_Category)s, %(Primary_Phrases)s, %(Secondary_Phrases)s,
                    %(Inclusion_Phrases)s, %(Exclusion_Phrases)s, %(IAge)s, %(IGender)s, %(EAge)s, %(EGender)s,
                    %(Drug_embeddings)s, %(Trial_Phase_embeddings)s, %(Population_Segment_embeddings)s,
                    %(Disease_Category_embeddings)s, %(Primary_Phrases_embeddings)s, %(Secondary_Phrases_embeddings)s,
                    %(Inclusion_Phrases_embeddings)s, %(Exclusion_Phrases_embeddings)s, %(IAge_embeddings)s,
                    %(IGender_embeddings)s, %(EAge_embeddings)s, %(EGender_embeddings)s
                );
            """
            # Combine text data and embeddings into a single dictionary
            data = {col: row[col] for col in df.columns if col not in embeddings}
            data.update(embeddings)
            cursor.execute(query, data)
        conn.commit()
    except mysql.connector.Error as err:
        print(f"Error: {err}")
    finally:
        conn.close()

# Function to generate embeddings for a list of texts
def get_batch_embeddings(text_list, batch_size=16):
    """
    Generate embeddings for a batch of text inputs.
    """
    embeddings = []
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i + batch_size]
        batch = [str(text) if pd.notna(text) else "unknown" for text in batch]

        inputs = tokenizer(batch, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1)  # Average over tokens
        embeddings.append(batch_embeddings)
    return torch.cat(embeddings, dim=0)

# Function to read data from an Excel file
def read_data_from_excel(file_path):
    """
    Read and preprocess data from an Excel file.
    """
    df = pd.read_excel(file_path)
    df.fillna('unknown', inplace=True)  # Replace NaN or missing values
    return df

