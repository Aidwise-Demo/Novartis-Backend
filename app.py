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


@app.post("/api/novartis/trial_details")
async def get_trial_details(request: Request):
    # Parse the JSON payload from the request body
    payload = await request.json()
    nctNumber = payload.get("nctNumber")

    # Handle the case where nctNumber is "not available"
    if not nctNumber or nctNumber.lower() == "not available":
        empty_trial_details = {
            "studyTitle": "",
            "primaryOutcomeMeasures": "",
            "secondaryOutcomeMeasures": "",
            "inclusionCriteria": "",
            "exclusionCriteria": ""
        }
        return JSONResponse(content={"trialDetails": [empty_trial_details]})

    # Connect to the database
    conn = get_db_connection()
    if conn is None:
        raise HTTPException(status_code=503, detail="Database connection failed.")

    try:
        # Query the database for trial details
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


@app.post("/api/novartis/top_trials")
async def get_top_trials(request: Request):
    # Parse the JSON payload from the request body
    payload = await request.json()

    # Extract parameters from the payload
    nctNumber = payload.get("nctNumber")
    studyTitle = payload.get("studyTitle")
    primaryOutcomeMeasures = payload.get("primaryOutcomeMeasures")
    secondaryOutcomeMeasures = payload.get("secondaryOutcomeMeasures")
    inclusionCriteria = payload.get("inclusionCriteria")
    exclusionCriteria = payload.get("exclusionCriteria")

    # Ensure at least one argument is provided
    if all(arg is None for arg in [
        studyTitle,
        primaryOutcomeMeasures,
        secondaryOutcomeMeasures,
        inclusionCriteria,
        exclusionCriteria
    ]):
        raise ValueError("At least one argument must be provided.")

    # Fetch data
    df = trials_extraction(
        nctNumber,
        studyTitle,
        primaryOutcomeMeasures,
        secondaryOutcomeMeasures,
        inclusionCriteria,
        exclusionCriteria
    )

    conn = get_db_connection()
    if conn is None:
        raise HTTPException(status_code=503, detail="Database connection failed.")

    try:
        # Convert column names to camelCase
        df.columns = [
            "nctNumber", "studyTitle", "primaryOutcomeMeasures", "secondaryOutcomeMeasures",
            "inclusionCriteria", "exclusionCriteria", "disease", "drug", "drugSimilarity",
            "inclusionCriteriaSimilarity", "exclusionCriteriaSimilarity",
            "studyTitleSimilarity", "primaryOutcomeMeasuresSimilarity",
            "secondaryOutcomeMeasuresSimilarity", "overallSimilarity"
        ]

        # Convert DataFrame to a list of dictionaries
        trials_list = df.to_dict(orient="records")
        response = json.dumps(trials_list)

        # Insert into the database
        insert_db(
            nctNumber, studyTitle, primaryOutcomeMeasures, secondaryOutcomeMeasures,
            inclusionCriteria, exclusionCriteria, response, conn
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
    finally:
        conn.close()

@app.post("/api/novartis/particular_trial")
async def get_particular_trial(request: Request):
    # Parse the JSON payload from the request body
    payload = await request.json()

    # Extract `nctNumber` from the payload
    nctNumber = payload.get("nctNumber")
    if not nctNumber:
        raise HTTPException(status_code=400, detail="nctNumber is required.")

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

        # Assuming response is in JSON-like format in the 'response' column
        response_data = df_saved['response'].iloc[0]  # Get the first response (most recent)

        # Deserialize the JSON string into a Python list of dictionaries
        try:
            trials_list = json.loads(response_data)
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="Invalid JSON format in the database.")

        # Ensure `trials_list` is a list
        if not isinstance(trials_list, list):
            raise HTTPException(status_code=500, detail="Response data is not a valid list of trials.")

        # Filter trials matching the given `nctNumber`
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
                "disease": trial.get("disease", "Not Available"),
                "studyTitle": trial.get("studyTitle", "Not Available"),
                "primaryOutcomeMeasures": trial.get("primaryOutcomeMeasures", "Not Available"),
                "secondaryOutcomeMeasures": trial.get("secondaryOutcomeMeasures", "Not Available"),
                "inclusionCriteria": trial.get("inclusionCriteria", "Not Available"),
                "exclusionCriteria": trial.get("exclusionCriteria", "Not Available"),
                "drug": trial.get("drug", "Not Available"),
                "studyTitleSimilarity": trial.get("studyTitleSimilarity", 0.0),
                "primaryOutcomeMeasuresSimilarity": trial.get("primaryOutcomeMeasuresSimilarity", 0.0),
                "secondaryOutcomeMeasuresSimilarity": trial.get("secondaryOutcomeMeasuresSimilarity", 0.0),
                "inclusionCriteriaSimilarity": trial.get("inclusionCriteriaSimilarity", 0.0),
                "exclusionCriteriaSimilarity": trial.get("exclusionCriteriaSimilarity", 0.0),
                "drugSimilarity": trial.get("drugSimilarity", 0.0),
                "overallSimilarity": trial.get("overallSimilarity", 0.0)
            }
            for trial in filtered_trials
        ]

        return JSONResponse(content={"trialDetails": result})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

    finally:
        if conn:
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
