import os
import tempfile
import json
import spacy
import logging
import numpy as np
import pandas as pd
from io import StringIO
from typing import Dict

from fuzzywuzzy import fuzz
from pydantic import BaseModel, ValidationError
from fastapi.exceptions import RequestValidationError, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi import FastAPI, Request, Body, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from starlette.background import BackgroundTask

from WrappedLLM import Initialize as ini, Output
from WrappedLLM.LLMModels import LLM_MODELS, get_info
from API.EntityExtractionModels import SingleEntityExtraction as SEE
from API.HandleResponses import Extraction as E, ChatGPT

# Load SpaCy model
nlp = spacy.load('en_core_web_md')

# Define confidence score weights
weights = {
    'exact_match': 0.3,
    'fuzzy_match': 0.45, 
    'spaCy_match': 0.25
}

# Initialize FastAPI
app = FastAPI()

# Allow all origins for demonstration purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your client's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=JSONResponse)
async def read_form():
    return {"message": "Single Entity Extraction Server is running"}


# Define the API endpoint
@app.post("/extract_entities")
async def extract_entities(file: UploadFile = File(...), ExtractionConfig: str = Form(...)):
    """
        Extracts entities from a CSV file using the specified LLM settings and target entities. The function takes a CSV file and an ExtractionConfig JSON string as input, and returns a BatchResponse containing the extracted entities.
        
        The function performs the following steps:
        1. Parses the ExtractionConfig JSON string to extract the LLM settings and target entities.
        2. Validates the input file, ensuring it is a non-empty CSV file.
        3. Reads the CSV file content into a pandas DataFrame.
        4. Initializes the appropriate LLM provider (OpenAI, Anthropic, or Google) based on the LLM settings.
        5. Iterates through the rows of the input DataFrame, extracting entities for each text using the specified LLM model.
        6. Returns a BatchResponse containing the extracted entities.
        
        The function handles various exceptions, including JSON decoding errors, validation errors, and file reading errors, and returns appropriate HTTP error responses.
    """
    
    API_KEY = ""
    MODEL_NAME = ""
    
    try:
        batch_request = SEE.Input.ExtractionConfig(**json.loads(ExtractionConfig))
        LLM_SETTINGS = batch_request.llm_settings
        TARGET_ENTITIES = batch_request.target_entities
        OUTPUT_FORMAT = batch_request.output_instructions
    
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON data: {e}")
    
    except ValidationError as e:
        raise RequestValidationError(e.errors())
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Validation error: {e}")    
    
    if file and file.size == 0:
        raise HTTPException(status_code=400, detail="Uploaded File is empty")
    
    # Check that the file is a CSV
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="The file must be a CSV")

    # Read the file content
    contents = await file.read()
    try:
        # Read the CSV data into a DataFrame
        input_data = pd.DataFrame(pd.read_csv(StringIO(contents.decode('utf-8')), index_col=False))
        input_data = input_data.dropna(subset=['Text'])
   
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV file: {e}")
    
    try:
        OUTPUT_OPTION = 'cont_cost'
        API_KEY = str(LLM_SETTINGS.api_key)
        MODEL_NAME = LLM_MODELS[LLM_SETTINGS.llm_provider][LLM_SETTINGS.llm_name]   
        
        if LLM_SETTINGS.llm_provider == 'openai':
            
            return ChatGPT.handleGPTResponse(input_data=input_data, API_KEY = API_KEY, TARGET_ENTITIES=TARGET_ENTITIES, OUTPUT_FORMAT=OUTPUT_FORMAT, MODEL_NAME=MODEL_NAME, LLM_SETTINGS=LLM_SETTINGS, OUTPUT_OPTION=OUTPUT_OPTION)
            
        if LLM_SETTINGS.llm_provider == 'anthropic':
            ini.init_claude(API_KEY)
            print("Claude initialized")
        
        if LLM_SETTINGS.llm_provider == 'google':
            ini.init_gemini(API_KEY)
            print("Gemini initialized")
        
        extracted_entities = []
        for index, item in input_data.iterrows():
            print("Extracting entity for:", item.text)
            
            entity = E.extract_entity(item.text, model = MODEL_NAME, settings=LLM_SETTINGS)
            extracted_entities.append(SEE.Output.ExtractedEntity(key=item.key, text=item.text, entity=entity))
        
        return SEE.Output.BatchResponse(extracted_entities=extracted_entities)
    
    except Exception as e:
        # raise HTTPException(status_code=500, detail={"message":"Invalid API Key!","error":str(e)})
        raise Exception(e)

@app.get("/available_models", response_model=Dict[str, Dict])
async def get_available_models(input_data: SEE.Input.AvailableModelsInput = Body(...)):
    if input_data.llm_provider == "All":
        # return {input_data.llm_provider: LLM_MODELS[input_data.llm_provider]}
        return JSONResponse(content=get_info())
        
    return JSONResponse(content=get_info(input_data.llm_provider))

@app.post("/entity_confidence")
async def calculate_entity_confidence(file: UploadFile = File(...)):
    """
    Calculate confidence scores for extracted entities in the input CSV.
    
    Expected CSV format:
    - Serial_No (int): Row identifier
    - Input_Text (str): Source text
    - Extracted_Entity (str): Entity to evaluate confidence for
    
    Returns:
    - CSV file with added Confidence_Score(%) column
    """
    
    # Input validation
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
        
    # Read CSV content
    try:
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode('utf-8')))
        
        # Validate required columns
        required_cols = ['Serial_No', 'Input_Text']
        if not all(col in df.columns for col in required_cols):
            raise HTTPException(
                status_code=400, 
                detail=f"CSV must contain columns: {', '.join(required_cols)}"
            )
        if len(df.columns.to_list()) > 3:
            raise HTTPException(
                status_code=400,
                detail=f"CSV must contain only columns: {', '.join(required_cols)}, Generic Entity Column (Any Name)"
            )
        
        fileName = df.columns.to_list()[-1]    
        
        # Compute confidence scores
        df['Confidence_Score(%)'] = df.apply(compute_confidence, axis=1)
        
        # Handle NaN values
        df.fillna("NaN", inplace=True)
        
        # Create temporary file for response
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
            df.to_csv(temp_file.name, index=False)
        
        def cleanup():
            try:
                os.unlink(temp_file.name)
            except:
                pass
                    
        # Return the file response
        return FileResponse(
            temp_file.name,
            media_type='text/csv',
            filename=f'{fileName}_with_confidence.csv',
            background=BackgroundTask(cleanup)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

def compute_confidence(row):
    """
    Compute confidence score for a single row using multiple matching techniques.
    
    Args:
        row (pd.Series): DataFrame row containing Input_Text and extracted entity
        
    Returns:
        float: Confidence score as percentage (0-100)
    """
    
    # Extract text and entity
    text = row['Input_Text']
    entity = row.iloc[-1] if 'Extracted_Entity' not in row else row['Extracted_Entity']
    
    # Handle NaN entities
    if pd.isna(entity):
        return 0
        
    # Convert to strings and lowercase
    text = str(text).lower()
    entity = str(entity).lower()
    
    # Calculate match scores
    exact_match = 1 if entity in text else 0
    fuzzy_match_score = fuzz.partial_ratio(entity, text) / 100
    
    # Calculate SpaCy similarity
    doc_entity = nlp(entity)
    doc_text = nlp(text)
    spaCy_match = doc_text.similarity(doc_entity)
    
    # Calculate weighted confidence score
    confidence_score = (
        weights['exact_match'] * exact_match +
        weights['fuzzy_match'] * fuzzy_match_score +
        weights['spaCy_match'] * spaCy_match
    )
    
    return round(confidence_score * 100, 2)

# Define error handler for validation errors
@app.exception_handler(RequestValidationError)
async def exception_handler(request: Request, exc: RequestValidationError):
    
    print(exc)
    
    ERROR_ROUTE = ERROR_TYPE = ERROR_MESSAGE = ERROR_FIELD = ERROR_INPUT = None
    
    try:
        errors = exc.errors()
        error_details = errors[0] if errors else {}
        
        ERROR_ROUTE = request.url.path
        ERROR_TYPE = error_details.get('type', None)
        
        if ERROR_TYPE == 'string_type' and len(errors) > 1:
            error_details = errors[1]
            ERROR_TYPE = error_details.get('type', None)
        
        ERROR_MESSAGE = error_details.get('msg', None)
        ERROR_FIELD = error_details.get('loc', [None, None])[-1] if len(error_details.get('loc', [])) > 0 else error_details.get('loc', [None])[0]
        ERROR_INPUT = error_details.get('input', None)
    
    except Exception as e:
        print("error in EXC, for exception_handler: ", str(e))
        # print(ERROR_TYPE, ERROR_MESSAGE, ERROR_FIELD, ERROR_ROUTE)
        ERROR_TYPE = ERROR_MESSAGE = ERROR_FIELD = ERROR_ROUTE = None
   
    if ERROR_ROUTE == "/available_models":
        
        if ERROR_TYPE == 'missing':
            return JSONResponse(
                status_code=404,
                content={"error": "Invalid Key, Did you mean: 'llm_provider'?"}
            )
        
        elif ERROR_TYPE == 'value_error':
            return JSONResponse(
                status_code=404,
                content={"error": ERROR_MESSAGE,"supported_providers": list(LLM_MODELS.keys()),"note": r"To view all supported providers, set 'llm_provider' to 'All' "}
            )
        
        else:
            return JSONResponse(
                status_code=404,
                content={"erroneous_field": ERROR_FIELD,"error": ERROR_MESSAGE}
            )
    
    if ERROR_ROUTE == "/extract_entities":
        
        if ERROR_TYPE == 'missing':
            if ERROR_FIELD not in ['target_entity','instructions','data']:
                return JSONResponse(
                    status_code=404,
                    content={"error": f"Invalid Key, Did you mean: '{ERROR_FIELD}' ?"}
                )
            else:
                if len(error_details['loc'])<=2:
                    return JSONResponse(
                        status_code=404,
                        content={"error": f"Invalid Key, Did you mean: '{ERROR_FIELD}' ?"}
                    )
                else:
                    return JSONResponse(
                        status_code=404,
                        content={"erroneous_field": ERROR_FIELD,"erroneous_input": ERROR_INPUT,"error": f"Invalid Key, Did you mean: '{error_details['loc'][-1]}' ?"}
                )
        
        elif ERROR_TYPE == 'value_error':
            
            if ERROR_FIELD == 'llm_provider':
                return JSONResponse(
                    status_code=404,
                    content={"erroneous_field": ERROR_FIELD,"error": ERROR_MESSAGE,"supported_providers": list(LLM_MODELS.keys())}
                )
            elif ERROR_FIELD == 'llm_name':
                
                error_ctx = error_details.get('ctx', {})
                error_value = error_ctx.get('error', {})

                if isinstance(error_value, ValueError):
                    
                    error_dict = error_value.args[0]
                    ERROR_MESSAGE = error_dict['message']
                    SUPPPORTED_MODELS = error_dict.get('supported_models', [])
                    
                    return JSONResponse(
                        status_code=404,
                        content={"erroneous_field": ERROR_FIELD,"error": ERROR_MESSAGE,"supported_models": SUPPPORTED_MODELS}
                    )
            elif ERROR_FIELD in ['instructions', 'target_entity']:
                
                return JSONResponse(
                        status_code=404,
                        content={"erroneous_field": f"'{error_details['loc'][-1]}' in 'output_{ERROR_FIELD}' at index '{error_details['loc'][-2]}'","erroneous_input": ERROR_INPUT,"error": ERROR_MESSAGE}
                )

            else:
                return JSONResponse(
                    status_code=404,
                    content={"erroneous_field": ERROR_FIELD,"error": ERROR_MESSAGE}
                )
        
        else:
            
            if ERROR_FIELD != 'data':
                
                return JSONResponse(
                    status_code=404,
                    content={"erroneous_field": ERROR_FIELD,"error": ERROR_MESSAGE}
                )
            else:
                return JSONResponse(
                        status_code=404,
                        content={"erroneous_field": f"'{error_details['loc'][-1]}' in '{ERROR_FIELD}' at index '{error_details['loc'][-2]}'","erroneous_input": ERROR_INPUT,"error": ERROR_MESSAGE}
                )
                  
    return JSONResponse(
        status_code=404,
        content={"detail": exc.errors()[0]},
    )
