# Standard library imports
import csv
import json
import logging
import os
import tempfile
import threading
from datetime import datetime
from io import StringIO

# Third-party imports
import mysql.connector
import pandas as pd
import requests
from mysql.connector import Error
from WrappedLLM import Output, Initialize as ini
from WrappedLLM.LLMModels import LLM_MODELS

# Logger configuration
logger = logging.getLogger('StudyTitleExtraction')
logger.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('\033[1m(%(levelname)s)\033[0m \033[1m[%(asctime)s]\033[0m \033[1m[Thread: %(threadName)s]\033[0m [%(funcName)s(Line: %(lineno)d)]: %(message)s', 
                            datefmt='%Y-%m-%d %H:%M:%S')

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Create file handler
file_handler = logging.FileHandler('study_title_extraction.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

API_KEY = os.getenv("OPENAI_API_KEY_MAYANK_AIDWISE_DEMO")

def get_prompt(identifier):
    
    global API_KEY
    
    prompts = {
        'disease_classification': {
            "system_prompt": """You are a knowledgeable assistant with access to a wide range of clinical trial information. When given a clinical trial title from the NCT database, classify it into a list of provided diseases based on the content and focus of the trial. Only return the disease title from the provided list, if the trial is classified as a disease. The format of the returned disease must strictly match the one in the list. If the trial cannot be classified, return "NaN". No other text should be included in the response.""",
            "user_prompt": "Given the NCT clinical trial title: \"{trial_title}\", classify it into the following diseases: {disease_list}. What disease(s) are associated with this trial?",
            },
        
        'study_title_entity_extraction': {
            "output_instructions": [
                    {
                        "target_entity": "Disease",
                        "instructions": "A disease is a condition disrupting the normal body function, marked by symptoms, signs, or anatomical changes. Diseases can be infectious (e.g., COVID-19), chronic (e.g., diabetes), or localized (e.g., esophageal cancer). Identify diseases by terms describing health issues, linked to anatomy (e.g., esophageal cancer) or with suffixes like -itis (inflammation), -osis (degenerative), or -emia (blood condition). Look for mentions of symptoms, risks, or abnormalities."
                    },
                    {
                        "target_entity": "Drug",
                        "instructions": "A drug is a chemical or biological substance used to diagnose, treat, or prevent diseases (e.g., ibuprofen, CLS2702C). Drugs can be identified by alphanumeric codes or suffixes like -mab (antibodies), -pril (ACE inhibitors), or -vir (antivirals). Key phrases like administered, treatment with, or inhibits often indicate drug mentions. They are commonly described in the context of their therapeutic purpose or evaluation in clinical studies."
                    },
                    {
                        "target_entity": "Trial Phase",
                        "instructions": "Refers to the clinical trial stage (e.g., Phase 1, Phase 2, or Phase 3). It often indicates the study's focus, such as safety testing (Phase 1), assessing efficacy (Phase 2), or large-scale effectiveness (Phase 3). For example, in A Phase 3 Study of Metformin, the trial phase is Phase 3. If no phase is mentioned in the title, it should be labeled as Not Specified."
                    },
                    {
                        "target_entity": "Population Segment",
                        "instructions": "Refers to the specific group being studied, such as Children with Asthma or Elderly Patients with Hypertension. It describes participants based on demographics, health status, or other defining characteristics. For example, in A Study of Drug X in Adolescents with Hypertension, the population segment is Adolescents with Hypertension. If no specific demographic or subgroup is mentioned, as in Phase II Iressa + Irradiation Followed by Chemo in NSCLC, it should be labeled as Not Specified."
                    }
                ],
                "target_entities": [
                    "Disease",
                    "Drug",
                    "Trial Phase",
                    "Population Segment"
                ],
                "llm_settings": {
                        "api_key": API_KEY,
                        "llm_provider": "openai",
                        "llm_name": "gpt4_omni_mini",
                        "max_tokens": 4096,
                        "temperature": 0.1,
                        "user_prompt": "Analyze the following medical text and extract the specified entities. Focus on identifying the following categories: diseases (e.g., esophageal cancer, diabetes), drug names (e.g., ibuprofen, CLS2702C), and their contextual details. Provide the extracted information in a structured and categorized format. If multiple items exist within a category, separate them using the | pipe symbol.",
                        "system_prompt": "You are a highly specialized language model trained to extract and categorize medical entities from text. Your task is to identify and organize the following categories of information:Diseases: Extract conditions or pathologies, such as esophageal cancer or diabetes, including terms with specific prefixes/suffixes like -itis, -osis, or -emia. Look for references to anatomical locations and symptoms for context. Drug Names: Identify chemical or biological substances, such as ibuprofen or CLS2702C, used for diagnosis, treatment, or prevention. Include drugs with alphanumeric codes or suffixes like -mab, -pril, or -vir. Your output should be concise, well-structured, and easy to interpret. For multiple extractions in a single category, separate them with the | pipe symbol to maintain clarity.",
                        "batch_size": 10
                }
            },
        
        'disease_details': {
            "output_instructions": "Classify each input term describing {disease_name}-related conditions into one of the following {num_categories} categories based on the provided examples and their clinical context: {categories}. Ensure that terms outside the provided examples are classified into the most relevant category based on their description and clinical context. If a term does not match any of the categories, return 'NaN'. If multiple {disease_name}-related instances are mentioned in a single input, select the most appropriate category based on the clinical context. The output should include the full category name as provided.",
            "target_entities": "{disease_name}_Category",
            "llm_settings": {
                "api_key": API_KEY,
                "llm_provider": "openai",
                "llm_name": "gpt4_omni",
                "max_tokens": 8192,
                "temperature": 0.15,
                "user_prompt": "Classify each {disease_name}-related term into one of the following {num_categories} categories based on the provided examples and their clinical context:\n\n{category_examples}\n\nEnsure that terms outside the provided examples are classified into the most relevant category based on their description and clinical context. If a term does not fit into any of these categories, return 'NaN.' In cases where multiple {disease_name}-related instances are mentioned in a single input, select the most appropriate category based on the clinical context. Provide the full category name as output.",
                "system_prompt": "You are a highly skilled medical language model specializing in {disease_name} classification. Your task is to classify {disease_name}-related terms into one of the following {num_categories} categories:\n\n{category_examples}\n\nElements outside the provided examples must be handled and classified appropriately into the most relevant category based on their clinical context and description. If a term does not match any category, return 'NaN.' In cases where multiple {disease_name}-related instances are mentioned in a single input, select the most appropriate category based on the clinical context. The output must include the full category name as provided."
            }
        },
        
        'study_design': """Extract the study design elements including phase, 
                          intervention model, and primary purpose.""",
                          
        'endpoints': """Identify and list all primary and secondary endpoints 
                       mentioned in the study."""
    }
    
    return prompts.get(identifier, "Prompt not found for the given identifier")

def query_SEE_endpoint(extraction_config: dict, study_title: str = None, disease: str = None) -> dict:
    
    if study_title and disease:
        logger.error("Both study_title and disease are provided. Please provide only one.")
    
    if study_title:
        logger.info(f"Extracting Entities from study title: {study_title}")
    if disease:
        logger.info(f"Categorising Disease: {disease}")
    
    # Create temporary CSV file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', newline='') as temp_csv:
        writer = csv.writer(temp_csv)
        writer.writerow(['Key', 'Text'])  # Header
        writer.writerow([1, study_title if study_title else disease])  # Data row
        temp_csv_path = temp_csv.name
    
    logger.info(f"Created temporary CSV at: {temp_csv_path}")
    
    endpoint_url = os.getenv("SEE_ENDPOINT_URL_AIDWISE_DEMO")
    
    # Prepare the files and form data
    files = {
        'file': ('input.csv', open(temp_csv_path, 'rb'), 'text/csv'),
        'ExtractionConfig': (None, json.dumps(extraction_config), 'application/json')
    }
    
    try:
        response = requests.post(
            url=endpoint_url,
            files=files,
        )
        
        response.raise_for_status()
        logger.info("Successfully received response from SEE endpoint")
        
        # Check content type of response
        content_type = response.headers.get('Content-Type', '')
        
        if 'application/json' in content_type:
            # Handle JSON response (likely error)
            result = response.json()
            logger.warning(f"Received JSON response (possible error): {result}")
            return result
        else:
            # Handle CSV response
            csv_data = StringIO(response.content.decode('utf-8'))
            df = pd.read_csv(csv_data)
            result_dict = df.to_dict(orient='records')

            del result_dict[0]['Serial_No']
            del result_dict[0]['Input_Text']
            
            logger.info(f"Successfully processed CSV response with {len(result_dict)} records")
            return result_dict[0]
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error occurred while querying SEE endpoint: {str(e)}")
        raise
    finally:
        files['file'][1].close()
        os.unlink(temp_csv_path)

def execute_query(query):
    logger.info(f"Executing query: {query}")
    try:
        connection = mysql.connector.connect(
            host=os.getenv('AIVENCLOUD_HOST_AIDWISE_DEMO'),
            user=os.getenv('AIVENCLOUD_USERNAME_AIDWISE_DEMO'),
            password=os.getenv('AIVENCLOUD_PASSWORD_AIDWISE_DEMO'),
            database=os.getenv('AIVENCLOUD_DATABASE_AIDWISE_DEMO'),
            port=os.getenv('AIVENCLOUD_PORT_AIDWISE_DEMO')
        )
        
        cursor = connection.cursor(dictionary=True)
        cursor.execute(query)
        results = cursor.fetchall()
        
        cursor.close()
        connection.close()
        
        logger.info(f"Query executed successfully, returned {len(results)} results")
        return results
        
    except Error as e:
        logger.error(f"Database error occurred: {str(e)}")
        return {"error": str(e)}

def extract_study_title_entities(study_title: str):
    
    global API_KEY
    logger.info(f"Starting study title extraction for: {study_title}")
    
    def classify_disease(study_title: str) -> str:
        
        logger.info("Classifying Disease from Study Title")
        prompt = get_prompt("disease_classification")
        uniqueDiseases = execute_query("SELECT distinct disease FROM clinicalstudy.conditions;")
        diseaseList ='[' + '|'.join(item['disease'] for item in uniqueDiseases) + ']'
        
        logger.info("Initializing LLM")
        if not ini.is_chatgpt_initialized():
            ini.init_chatgpt(api_key=API_KEY)
        
        logger.info("Running LLM classification")
        classifiedDisease = Output.GPT(
            user_prompt=prompt["user_prompt"].format(trial_title=study_title, disease_list=diseaseList),
            system_prompt=prompt["system_prompt"],
            model=LLM_MODELS.get("openai").get("gpt4_omni_mini"),
            output_option='cont'
        )
        if classifiedDisease != "NaN":
            logger.info(f"Disease classification completed: {classifiedDisease}")
            return classifiedDisease
        return ""
    
    def getDiseaseDetailsPrompt(classifiedDisease: str, diseaseDetails: dict) -> str:
        
        logger.info("Generating Disease Details Prompt...")
        diseaseDetailsPrompt = get_prompt("disease_details")
        
        categories = [entry['Disease_Category'] for entry in diseaseDetails]
        examples = [entry['Examples'] for entry in diseaseDetails]
        examples="\n".join([f'[{i+1}] {categories[i]}: {examples[i]}' for i in range(len(categories))])
        categories=', '.join([f'[{i+1}] {categories[i]}' for i in range(len(categories))])
        
        formatted_target_entities = diseaseDetailsPrompt["target_entities"].format(disease_name=classifiedDisease)
        formatted_output_instructions = diseaseDetailsPrompt["output_instructions"].format(
            disease_name=classifiedDisease,
            num_categories=len(diseaseDetails),
            categories=categories,
        )
        formatted_system_prompt = diseaseDetailsPrompt["llm_settings"]["system_prompt"].format(
            disease_name=classifiedDisease,
            num_categories=len(diseaseDetails),
            category_examples=examples
        )
        formatted_user_prompt = diseaseDetailsPrompt["llm_settings"]["user_prompt"].format(
            disease_name=classifiedDisease,
            num_categories=len(diseaseDetails),
            category_examples=examples
        )

        # Update the formatted values in the prompt dictionary
        diseaseDetailsPrompt["target_entities"] = formatted_target_entities
        diseaseDetailsPrompt["output_instructions"] = formatted_output_instructions
        diseaseDetailsPrompt["llm_settings"]["system_prompt"] = formatted_system_prompt
        diseaseDetailsPrompt["llm_settings"]["user_prompt"] = formatted_user_prompt
        
        return diseaseDetailsPrompt
    
    def getDiseaseDetails(extractedDisease:str) -> dict:
        
        classifiedDisease = classify_disease(study_title=study_title)
        
        if classifiedDisease:
            
            logger.info(f"Disease found in study title: {classifiedDisease}")
            diseaseDetails = execute_query(f"SELECT Disease_Category, Examples FROM clinicalstudy.diseasecategory WHERE Disease = '{classifiedDisease}';")
            
            diseaseDetailsPrompt = getDiseaseDetailsPrompt(classifiedDisease=classifiedDisease, diseaseDetails=diseaseDetails) 
            categorisedDisease = query_SEE_endpoint(disease=extractedDisease, extraction_config=diseaseDetailsPrompt) 

            return {"Disease":classifiedDisease,"Disease_Category": categorisedDisease.get(f"{classifiedDisease}_Category")}
        
        else: 
            logger.warning("No disease found in study title")
            return {"Disease": None,"Disease_Category": None} 

    studyTitleEntities = query_SEE_endpoint(study_title=study_title, extraction_config=get_prompt("study_title_entity_extraction"))
    studyTitleEntities["Primary_Disease"] = studyTitleEntities.pop('Disease')
    finalResult = getDiseaseDetails(extractedDisease=studyTitleEntities['Primary_Disease'])
    finalResult['Study_Title_Entities'] = studyTitleEntities
    finalResult["Study_Title"] = study_title
    
    return finalResult
#
# studyTitle: str =  "Safety and Efficacy of Deep Wave Trabeculoplasty (DWT) in Primary Open Angle Glaucoma and Ocular Hypertension"
# # # studyTitle: str = "A Safety and Efficacy Study of MSI-1256F (Squalamine Lactate) To Treat 'Wet' Age-Related Macular Degeneration"
# result = extract_study_title_entities(study_title=studyTitle)
# print(result)