# Clinical Trials Semantic-Based Similarity Scoring System

This repository implements a pipeline for processing clinical trial data, extracting relevant entities, tagging information, generating embeddings, and calculating similarity scores. The system is built using **FastAPI** and integrates various modules for database interaction, natural language processing, and similarity scoring.

---

## Table of Contents

1. [Project Overview](#project-overview) 
2. [Workflow Steps](#workflow-steps)
3. [Key Features](#key-features)  
4. [Technologies Used](#technologies-used)  
5. [Folder Structure](#folder-structure)  
6. [Setup Instructions](#setup-instructions)
7. [API Endpoints](#api-endpoints)  
8. [Output](#output)  
9. [Contributors](#contributors)  
10. [Future Enhancements](#future-enhancements)

---

## Project Overview

This project is designed to streamline the analysis and comparison of clinical trial data. The workflow involves multiple steps, from data retrieval to similarity scoring, using a modular and efficient pipeline.

---

## Workflow Steps

1. **Input Handling**  
   Frontend inputs (NCT Numbers) are received via FastAPI endpoints (`app.py`).

2. **Database Interaction**  
   Data such as Study Title, Primary/Secondary Outcome Measures, and Inclusion/Exclusion Criteria are retrieved using `db_data_retriever.py`.

3. **Entity Extraction**  
   Extracts entities such as **Drug**, **Disease**, **Population Segment**, and **Disease Category** from the Study Title using:
   - LLM integration (`llm_handler.py`)
   - Query execution via `query_executor.py`

4. **Tagging and Processing**  
   Keywords from the outcome measures and criteria are tagged using:  
   - `phrases_extractor.py`  
   - `phrases_tagging.py`  
   - `metadata_extraction.py`

5. **Embedding Generation**  
   Embeddings for tagged data are created using:
   - `embedding_generator.py`  
   - Processed by `embedding_processor.py`.

6. **Similarity Scoring**  
   - Similarities between clinical trials are calculated using:  
     - `find_similar_trials.py`  
     - `similarity_calculator.py`  
   - Scores are aggregated (`score_aggregation.py`) and normalized (`weight_normalization.py`).

7. **Result Storage and Response**  
   - Final results are stored in the database (`db_history_loader.py`).  
   - JSON responses are returned to the frontend.

---

## Key Features

- **Real-time Processing**: FastAPI enables real-time input handling and response generation.  
- **LLM Integration**: Uses Large Language Models for advanced entity extraction and tagging.  
- **Customizable Scoring**: Supports manual weight normalization for fine-tuned similarity scoring.  
- **Scalable Design**: Modular architecture for seamless scalability and maintenance.  
- **Data-Driven Insights**: Embedding-based similarity scoring ensures accurate comparisons.

---

## Technologies Used

- **Backend Framework**: FastAPI  
- **Database**: SQL-based (compatible with MySQL, PostgreSQL, etc.)  
- **LLM Framework**: Hugging Face Transformers / OpenAI API (as applicable)  
- **Embedding Library**: Sentence Transformers  
- **Similarity Scoring**: Cosine Similarity, Custom Aggregation  
- **Programming Language**: Python  

---

## Folder Structure

```plaintext
├── app.py                     # FastAPI app for handling inputs and APIs
├── database
│   ├── db_data_retriever.py   # Retrieves data from the database
│   ├── db_history_loader.py   # Saves processed data to the database
├── llm
│   ├── llm_handler.py         # LLM invocation and processing
├── extraction
│   ├── study_title_processing.py  # Entity extraction from Study Title
│   ├── metadata_extraction.py     # Metadata processing
├── tagging
│   ├── phrases_extractor.py   # Extracts phrases for tagging
│   ├── phrases_tagging.py     # Tags phrases with keywords
├── embeddings
│   ├── embedding_generator.py # Generates embeddings
│   ├── embedding_processor.py # Processes generated embeddings
├── similarities
│   ├── find_similar_trials.py     # Finds similar trials
│   ├── similarity_calculator.py  # Calculates similarity scores
├── scoring
│   ├── score_aggregation.py       # Aggregates similarity scores
│   ├── weight_normalization.py    # Normalizes weights
├── utils
│   ├── query_executor.py      # Executes LLM queries
└── README.md                  # Project documentation
```
---

## Setup Instructions
To set up the project locally:

1. **Install Python 3.12**  
   Ensure you have Python 3.12 installed on your system. You can download it from [here](https://www.python.org/downloads/release/python-3120/).

2. **Clone the repository**  
   Clone the repository to your local machine:
   ```bash
   git clone https://github.com/your-repository-url.git
   cd your-repository-folder

---

## API Endpoints
The following endpoints are available in the [API](https://api.novartis-backend.aidwise.in/):

- **GET `/api/novartis/nct_numbers`**: Retrieves NCT numbers.
- **POST `/api/novartis/trial_details`**: Submit trial details to be processed.
- **POST `/api/novartis/top_trials`**: Retrieve top trials based on certain criteria.
- **POST `/api/novartis/particular_trial`**: Retrieve details for a specific trial.
- **GET `/api/novartis/input_history`**: Retrieves the history of inputs made.
- **POST `/api/novartis/top_trials_nct`**: This endpoint is used when setting up the system locally to fetch top trials based on NCT.

---

## Output
- **For Host-based Output**: The output can be viewed in a browser by accessing the respective endpoint in the [API](https://api.novartis-backend.aidwise.in/).

- **For Web-based Output**: The output can be viewed in a website by accessing the [link](https://novartis.aidwise.in/).
  
- **For Local Setup**: The results are provided in an Excel file format. You can download the file containing trial details after processing through the relevant API endpoint.

---

## Contributors
This project is maintained by **Aidwise Analytics Pvt Ltd**. The contributors are:
- **Mayank Jalan**
- **Neha Luthra**
- **Sharath Suram**
- **Jayam Gupta**
- **Sidhant Raj**

----

## Future Enhancements
- Add support for additional trial data sources.
- Improve processing speed and optimize the database queries.
- Integrate a more advanced tagging mechanism for clinical trial keywords.
- Provide user authentication and authorization for API access.
