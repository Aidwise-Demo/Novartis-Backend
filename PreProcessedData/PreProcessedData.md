# Clinical Trial Dataset Pre-Processing and Embedding Workflow

## Overview
This project involves processing a clinical trial dataset to:

- Preprocess data by filtering and splitting conditions.
- Tag phrases in the dataset using disease-specific keywords.
- Generate embeddings for specified columns.
- Save processed data and embeddings to a database.

## Workflow

### Filter and Split Conditions
- The input dataset is filtered based on specific conditions.
- Split conditions are mapped to relevant disease categories.

### Preprocess Dataset
- Using the `Dataset` class, process and clean the data for further analysis.
- Ensure required columns are present and formatted correctly.

### Tag Keywords
- Disease-specific keywords are loaded and applied to tag relevant phrases in key columns (e.g., Primary Outcomes, Secondary Outcomes).
- Uses fuzzy matching to ensure partial matches are captured.

### Create Embeddings Table
- Check if the embeddings table exists in the database. If not, create it.

### Generate Embeddings
- For specified columns, generate embeddings using a pre-trained embedding generator.
- Add generated embeddings as new columns in the dataset.

### Save to Database
- Save the processed dataset with embeddings to the database for further use.

## Setup and Installation

### Install required libraries:
```bash
pip install pandas rapidfuzz openpyxl
