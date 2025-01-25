from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from database.db_history_loader import insert_db
from database.mysql_connector import get_db_connection
from Main import trials_extraction
import json
import pandas as pd

# Initialize the FastAPI application
app = FastAPI()

# CORS middleware configuration to allow requests from any origin
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware for DB connection handling (optional)
@app.middleware("http")
async def db_connection_middleware(request: Request, call_next):
    response = await call_next(request)
    return response

# Endpoint to fetch distinct NCT numbers
@app.get("/api/novartis/nct_numbers")
async def get_nct_number():
    conn = get_db_connection()
    if conn is None:
        raise HTTPException(status_code=503, detail="Database connection failed.")
    try:
        # Query the database to get distinct NCT numbers
        query = "SELECT DISTINCT(NCT_Number) FROM embedding"
        df_saved = pd.read_sql(query, conn)

        distinct_nct_numbers = df_saved['NCT_Number'].tolist()
        distinct_nct_numbers.append("Not Available")

        return JSONResponse(content={"nctNumbers": distinct_nct_numbers})
    finally:
        conn.close()

# Endpoint to fetch trial details based on NCT number
@app.post("/api/novartis/trial_details")
async def get_trial_details(request: Request):
    payload = await request.json()  # Parse the incoming JSON payload
    nctNumber = payload.get("nctNumber")

    if not nctNumber or nctNumber.lower() == "not available":
        # Return empty trial details if NCT number is not available
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
        # Query for trial details by NCT number
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

        # Convert database rows to a list of dictionaries and camelCase keys
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

# Endpoint to get top trials based on various parameters
@app.post("/api/novartis/top_trials")
async def get_top_trials(request: Request):
    payload = await request.json()  # Parse the incoming JSON payload

    # Extract parameters from payload
    nctNumber = payload.get("nctNumber")
    studyTitle = payload.get("studyTitle")
    primaryOutcomeMeasures = payload.get("primaryOutcomeMeasure")
    secondaryOutcomeMeasures = payload.get("secondaryOutcomeMeasure")
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
        raise HTTPException(status_code=400, detail="At least one argument must be provided.")

    # Call the trials_extraction function to get trial data
    try:
        result = trials_extraction(
            nctNumber,
            studyTitle,
            primaryOutcomeMeasures,
            secondaryOutcomeMeasures,
            inclusionCriteria,
            exclusionCriteria
        )
        # Check if result is a string (error message from the model)
        if isinstance(result, str):
            return JSONResponse(
                content={"message": result},  # Return the model's limitation or error message
                status_code=200
            )

        # Ensure result is a DataFrame
        if not isinstance(result, pd.DataFrame):
            raise ValueError("Unexpected result type from trials_extraction. Expected DataFrame.")

        conn = get_db_connection()
        if conn is None:
            raise HTTPException(status_code=503, detail="Database connection failed.")

        try:
            # Rename columns in the DataFrame for consistency
            result.columns = [
                "nctNumber", "studyTitle", "primaryOutcomeMeasures", "secondaryOutcomeMeasures",
                "inclusionCriteria", "exclusionCriteria", "disease", "drug", "drugSimilarity",
                "inclusionCriteriaSimilarity", "exclusionCriteriaSimilarity",
                "studyTitleSimilarity", "primaryOutcomeMeasuresSimilarity",
                "secondaryOutcomeMeasuresSimilarity", "overallSimilarity"
            ]

            # Convert the DataFrame to a list of dictionaries
            trials_list = result.to_dict(orient="records")

            # Insert the response into the database
            insert_db(
                nctNumber, studyTitle, primaryOutcomeMeasures, secondaryOutcomeMeasures,
                inclusionCriteria, exclusionCriteria, json.dumps(trials_list), conn
            )

            # Filter the result for a specific set of fields
            filtered_trials_list = [
                {
                    "nctNumber": trial["nctNumber"],
                    "studyTitle": trial["studyTitle"],
                    "overallSimilarity": trial["overallSimilarity"]
                }
                for trial in trials_list
            ]

            return JSONResponse(content={"trials": filtered_trials_list})
        finally:
            conn.close()

    except ValueError as e:
        return JSONResponse(
            content={"message": str(e)},
            status_code=400
        )

    except Exception as e:
        return JSONResponse(
            content={"message": "An unexpected error occurred. Please try again later.", "error": str(e)},
            status_code=500
        )

# Endpoint to fetch a specific trial based on NCT number from history data
@app.post("/api/novartis/particular_trial")
async def get_particular_trial(request: Request):
    payload = await request.json()  # Parse the incoming JSON payload

    nctNumber = payload.get("nctNumber")
    if not nctNumber:
        raise HTTPException(status_code=400, detail="nctNumber is required.")

    conn = get_db_connection()
    if conn is None:
        raise HTTPException(status_code=503, detail="Database connection failed.")

    try:
        # Query to fetch the latest trial data from history
        query = """
                SELECT response
                FROM history
                WHERE Serial_Number = (SELECT MAX(Serial_Number) FROM history);
        """
        df_saved = pd.read_sql(query, conn)

        if df_saved.empty:
            raise HTTPException(status_code=404, detail="No trial data found.")

        response_data = df_saved['response'].iloc[0]  # Get most recent response

        # Deserialize JSON string into a Python object
        trials_list = json.loads(response_data)

        # Ensure the trials list is valid
        if not isinstance(trials_list, list):
            raise HTTPException(status_code=500, detail="Response data is not a valid list of trials.")

        # Filter trials matching the provided NCT number
        filtered_trials = [
            trial for trial in trials_list if trial.get('nctNumber', '').lower() == nctNumber.lower()
        ]

        if not filtered_trials:
            raise HTTPException(status_code=404, detail="Trial not found for the given NCT number.")

        # Return the filtered trial data
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

# Endpoint to fetch history of trial inputs from the database
@app.get("/api/novartis/input_history")
async def get_history_input():
    conn = get_db_connection()
    if conn is None:
        raise HTTPException(status_code=503, detail="Database connection failed.")

    try:
        # Query to fetch the latest trial data from the history table
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
        df_saved = pd.read_sql(query, conn)

        if df_saved.empty:
            raise HTTPException(status_code=404, detail="No trial data found.")

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
