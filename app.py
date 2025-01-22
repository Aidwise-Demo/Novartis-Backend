from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from insertDB import insert_db
from Main import trials_extraction
from dotenv import load_dotenv
import mysql.connector
import os
import json
import pandas as pd

# Load environment variables from .env file
load_dotenv()

# Function to establish a MySQL connection
def get_db_connection():
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

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def db_connection_middleware(request: Request, call_next):
    response = await call_next(request)
    return response

@app.get("/api/novartis/nct_numbers")
async def get_nct_number():
    conn = get_db_connection()
    if conn is None:
        raise HTTPException(status_code=503, detail="Database connection failed.")
    try:
        query = "SELECT DISTINCT(NCT_Number) FROM embeddings"
        df_saved = pd.read_sql(query, conn)

        distinct_nct_numbers = df_saved['NCT_Number'].tolist()
        distinct_nct_numbers.append("Not Available")

        return JSONResponse(content={"nctNumbers": distinct_nct_numbers})
    finally:
        conn.close()


@app.get("/api/novartis/trial_details")
async def get_trial_details(nctNumber: str):
    if nctNumber.lower() == "not available":
        # Return an empty JSON structure with keys set to empty values
        empty_trial_details = {
            "studyTitle": "",
            "primaryOutcomeMeasures": "",
            "secondaryOutcomeMeasures": "",
            "inclusionCriteria": "",
            "exclusionCriteria": ""
        }
        return JSONResponse(content={"trialDetails": [empty_trial_details]})

    conn = get_db_connection()
    if conn is None:
        raise HTTPException(status_code=503, detail="Database connection failed.")

    try:
        query = """
            SELECT 
                Study_Title, 
                Primary_Outcome_Measures, 
                Secondary_Outcome_Measures, 
                Inclusion_Criteria, 
                Exclusion_Criteria 
            FROM clinicaltrials 
            WHERE LOWER(NCT_Number) = %s
        """
        df_saved = pd.read_sql(query, conn, params=(nctNumber.lower(),))

        # Map the DataFrame to a list of dictionaries with camelCase keys
        trial_details = df_saved.to_dict(orient='records')
        trial_details_camel_case = [
            {
                "studyTitle": record["Study_Title"],
                "primaryOutcomeMeasures": record["Primary_Outcome_Measures"],
                "secondaryOutcomeMeasures": record["Secondary_Outcome_Measures"],
                "inclusionCriteria": record["Inclusion_Criteria"],
                "exclusionCriteria": record["Exclusion_Criteria"]
            }
            for record in trial_details
        ]

        return JSONResponse(content={"trialDetails": trial_details_camel_case})
    finally:
        conn.close()


@app.get("/api/novartis/top_trials")
async def get_top_trials(
        NCT_Number: str = None,
        Study_Title: str = None,
        Primary_Outcome_Measures: str = None,
        Secondary_Outcome_Measures: str = None,
        Inclusion_Criteria: str = None,
        Exclusion_Criteria: str = None
):
    if all(arg is None for arg in [
        Study_Title,
        Primary_Outcome_Measures,
        Secondary_Outcome_Measures,
        Inclusion_Criteria,
        Exclusion_Criteria
    ]):
        raise ValueError("At least one argument must be provided.")

    # Fetch data
    df = trials_extraction(
        NCT_Number,
        Study_Title,
        Primary_Outcome_Measures,
        Secondary_Outcome_Measures,
        Inclusion_Criteria,
        Exclusion_Criteria
    )

    conn = get_db_connection()
    if conn is None:
        raise HTTPException(status_code=503, detail="Database connection failed.")

    # Convert column names to camelCase
    df.columns = [
        "nctNumber", "studyTitle", "primaryOutcomeMeasures", "secondaryOutcomeMeasures",
        "inclusionCriteria", "exclusionCriteria", "drug", "diseaseCategory",
        "populationSegment", "studyTitleSimilarity", "primaryOutcomeMeasuresSimilarity",
        "secondaryOutcomeMeasuresSimilarity", "inclusionCriteriaSimilarity",
        "exclusionCriteriaSimilarity", "drugSimilarity", "diseaseCategorySimilarity",
        "populationSegmentSimilarity", "overallSimilarity"
    ]

    # Convert DataFrame to a list of dictionaries
    trials_list = df.to_dict(orient="records")

    # Insert into the database
    insert_db(
        NCT_Number, Study_Title, Primary_Outcome_Measures, Secondary_Outcome_Measures,
        Inclusion_Criteria, Exclusion_Criteria, trials_list, conn
    )

    # Filter the results to return only nctNumber, studyTitle, and overallSimilarity
    filtered_trials_list = [
        {
            "nctNumber": trial["nctNumber"],
            "studyTitle": trial["studyTitle"],
            "overallSimilarity": trial["overallSimilarity"]
        }
        for trial in trials_list
    ]

    # Return the filtered result as JSON
    return JSONResponse(content={"trials": filtered_trials_list})

@app.get("/api/novartis/particular_trial")
async def get_particular_trial(nctNumber: str):
    conn = get_db_connection()
    if conn is None:
        raise HTTPException(status_code=503, detail="Database connection failed.")

    try:
        # Query to fetch the response with the latest Serial_Number
        query = """
                SELECT response
                FROM history
                WHERE Serial_Number = (SELECT MAX(Serial_Number) FROM history);
        """
        # Execute the query and fetch the response
        df_saved = pd.read_sql(query, conn)

        if df_saved.empty:
            raise HTTPException(status_code=404, detail="No trial data found.")

        # Assuming response is in a JSON-like format in the 'response' column
        trials_list = df_saved['response'].iloc[0]  # Get the first response (since it's the most recent)

        # If the response is a list of dictionaries (trials data)
        filtered_trials = [
            trial for trial in trials_list
            if trial.get('nctNumber', '').lower() == nctNumber.lower()
        ]

        if not filtered_trials:
            raise HTTPException(status_code=404, detail="Trial not found for the given NCT number.")

        # Return the filtered trials in the desired format
        result = [
            {
                "nctNumber": trial["nctNumber"],
                "studyTitle": trial["studyTitle"],
                "primaryOutcomeMeasures": trial["primaryOutcomeMeasures"],
                "secondaryOutcomeMeasures": trial["secondaryOutcomeMeasures"],
                "inclusionCriteria": trial["inclusionCriteria"],
                "exclusionCriteria": trial["exclusionCriteria"],
                "drug": trial["drug"],
                "diseaseCategory": trial["diseaseCategory"],
                "populationSegment": trial["populationSegment"],
                "studyTitleSimilarity": trial["studyTitleSimilarity"],
                "primaryOutcomeMeasuresSimilarity": trial["primaryOutcomeMeasuresSimilarity"],
                "secondaryOutcomeMeasuresSimilarity": trial["secondaryOutcomeMeasuresSimilarity"],
                "inclusionCriteriaSimilarity": trial["inclusionCriteriaSimilarity"],
                "exclusionCriteriaSimilarity": trial["exclusionCriteriaSimilarity"],
                "drugSimilarity": trial["drugSimilarity"],
                "diseaseCategorySimilarity": trial["diseaseCategorySimilarity"],
                "populationSegmentSimilarity": trial["populationSegmentSimilarity"],
                "overallSimilarity": trial["overallSimilarity"]
            }
            for trial in filtered_trials
        ]

        return JSONResponse(content={"trialDetails": result})

    finally:
        conn.close()

@app.get("/api/novartis/input_history")
async def get_history_input():
    conn = get_db_connection()
    if conn is None:
        raise HTTPException(status_code=503, detail="Database connection failed.")

    try:
        # Query to fetch the response with the latest Serial_Number
        query = """
                SELECT NCT_Number,
                Study_Title, 
                Primary_Outcome_Measures, 
                Secondary_Outcome_Measures, 
                Inclusion_Criteria, 
                Exclusion_Criteria 
                FROM history
                WHERE Serial_Number = (SELECT MAX(Serial_Number) FROM history);
        """
        # Execute the query and fetch the response
        df_saved = pd.read_sql(query, conn)

        if df_saved.empty:
            raise HTTPException(status_code=404, detail="No trial data found.")

        # Map the DataFrame to a list of dictionaries with camelCase keys
        trial_details = df_saved.to_dict(orient='records')
        trial_details_camel_case = [
            {
                "nctNumber": record["NCT_Number"],
                "studyTitle": record["Study_Title"],
                "primaryOutcomeMeasures": record["Primary_Outcome_Measures"],
                "secondaryOutcomeMeasures": record["Secondary_Outcome_Measures"],
                "inclusionCriteria": record["Inclusion_Criteria"],
                "exclusionCriteria": record["Exclusion_Criteria"]
            }
            for record in trial_details
        ]

        return JSONResponse(content={"trialDetails": trial_details_camel_case})
    finally:
        conn.close()
