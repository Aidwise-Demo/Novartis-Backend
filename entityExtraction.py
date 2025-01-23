# Standard library imports
import csv
import json
import logging
import os
import tempfile
import threading
from datetime import datetime
from io import StringIO
from typing import Dict, List, Union, Any, Optional

# Third-party imports
import pandas as pd
import requests
from dotenv import load_dotenv
import mysql.connector
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

load_dotenv()
apiKey = os.getenv("OPENAI_API_KEY_MAYANK_AIDWISE_DEMO")
endpointURL = os.getenv("SEE_ENDPOINT_URL_AIDWISE_DEMO")

class LLM:
    def __init__(self, apiKey: str):
        self.apiKey = apiKey


    def getPrompt(self, identifier: str) -> Dict[str,Union[str, Any]]:
        """
        Retrieves a dictionary of prompts for various natural language processing tasks, such as disease classification and entity extraction from medical text. The prompts include system and user prompts, output instructions, and LLM settings for the specified task.
        
            The `get_prompt` method takes an `identifier` parameter, which is a string that specifies the task to retrieve the prompt for. The method returns a dictionary containing the relevant prompts and settings for the specified task.     
        """
        prompts = {
            'diseaseClassification': {
                "systemPrompt": """You are a knowledgeable assistant with access to a wide range of clinical trial information. When given a clinical trial title from the NCT database, classify it into a list of provided diseases based on the content and focus of the trial. Only return the disease title from the provided list, if the trial is classified as a disease. The format of the returned disease must strictly match the one in the list. If the trial cannot be classified, return "NaN". No other text should be included in the response.""",
                "userPrompt": "Given the NCT clinical trial title: \"{trialTitle}\", classify it into the following diseases: {diseaseList}. What disease(s) are associated with this trial?",
                },
            
            'studyTitleEntityExtraction': {
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
                            "api_key": self.apiKey,
                            "llm_provider": "openai",
                            "llm_name": "gpt4_omni",
                            "max_tokens": 4096,
                            "temperature": 0.1,
                            "user_prompt": "Analyze the following medical text and extract the specified entities. Focus on identifying the following categories: diseases (e.g., esophageal cancer, diabetes), drug names (e.g., ibuprofen, CLS2702C), and their contextual details. Provide the extracted information in a structured and categorized format. If multiple items exist within a category, separate them using the | pipe symbol.",
                            "system_prompt": "You are a highly specialized language model trained to extract and categorize medical entities from text. Your task is to identify and organize the following categories of information:Diseases: Extract conditions or pathologies, such as esophageal cancer or diabetes, including terms with specific prefixes/suffixes like -itis, -osis, or -emia. Look for references to anatomical locations and symptoms for context. Drug Names: Identify chemical or biological substances, such as ibuprofen or CLS2702C, used for diagnosis, treatment, or prevention. Include drugs with alphanumeric codes or suffixes like -mab, -pril, or -vir. Your output should be concise, well-structured, and easy to interpret. For multiple extractions in a single category, separate them with the | pipe symbol to maintain clarity.",
                            "batch_size": 10
                    }
                },
            
            'diseaseDetails': {
                "output_instructions": "Classify each input term describing {diseaseName}-related conditions into one of the following {numCategories} categories based on the provided examples and their clinical context: {categories}. Ensure that terms outside the provided examples are classified into the most relevant category based on their description and clinical context. If a term does not match any of the categories, return 'NaN'. If multiple {diseaseName}-related instances are mentioned in a single input, select the most appropriate category based on the clinical context. The output should include the full category name as provided.",
                "target_entities": "{diseaseName}_Category",
                "llm_settings": {
                    "api_key": self.apiKey,
                    "llm_provider": "openai",
                    "llm_name": "gpt4_omni",
                    "max_tokens": 8192,
                    "temperature": 0.15,
                    "user_prompt": "Classify each {diseaseName}-related term into one of the following {numCategories} categories based on the provided examples and their clinical context:\n\n{categoryExamples}\n\nEnsure that terms outside the provided examples are classified into the most relevant category based on their description and clinical context. If a term does not fit into any of these categories, return 'NaN.' In cases where multiple {diseaseName}-related instances are mentioned in a single input, select the most appropriate category based on the clinical context. Provide the full category name as output.",
                    "system_prompt": "You are a highly skilled medical language model specializing in {diseaseName} classification. Your task is to classify {diseaseName}-related terms into one of the following {numCategories} categories:\n\n{categoryExamples}\n\nElements outside the provided examples must be handled and classified appropriately into the most relevant category based on their clinical context and description. If a term does not match any category, return 'NaN.' In cases where multiple {diseaseName}-related instances are mentioned in a single input, select the most appropriate category based on the clinical context. The output must include the full category name as provided."
                }
            },
        }
        
        return prompts.get(identifier, "Prompt not found for the given identifier")


    def querySEEEndpoint(self, extractionConfig: Dict[str,Any], studyTitle: str = None, disease: str = None) -> Dict[str, Any]:
        """
            Queries the SEE (Structured Entity Extraction) endpoint with the provided extraction configuration and either a study title or disease name.
            
            Args:
                extractionConfig (Dict[str, Any]): The extraction configuration to be used for the SEE endpoint.
                studyTitle (str, optional): The study title to extract entities from.
                disease (str, optional): The disease name to categorize.
            
            Returns:
                Dict[str, Any]: The response from the SEE endpoint, either as a JSON object or a dictionary of extracted entities.
        """
                
        if studyTitle and disease:
            logger.error("Both studyTitle and disease are provided. Please provide only one.")
        
        if studyTitle:
            logger.info(f"Extracting Entities from study title: {studyTitle}")
        if disease:
            logger.info(f"Categorising Disease: {disease}")
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', newline='') as tempCsv:
            writer = csv.writer(tempCsv)
            writer.writerow(['Key', 'Text'])
            writer.writerow([1, studyTitle if studyTitle else disease])
            tempCsvPath = tempCsv.name
        
        logger.info(f"Created temporary CSV at: {tempCsvPath}")
        
        endpointUrl = os.getenv("SEE_ENDPOINT_URL_AIDWISE_DEMO")
        
        files = {
            'file': ('input.csv', open(tempCsvPath, 'rb'), 'text/csv'),
            'ExtractionConfig': (None, json.dumps(extractionConfig), 'application/json')
        }
        
        try:
            response = requests.post(
                url=endpointUrl,
                files=files,
            )
            
            response.raise_for_status()
            logger.info("Successfully received response from SEE endpoint")
            
            contentType = response.headers.get('Content-Type', '')
            
            if 'application/json' in contentType:
                result = response.json()
                logger.warning(f"Received JSON response (possible error): {result}")
                return result
            else:
                csvData = StringIO(response.content.decode('utf-8'))
                df = pd.read_csv(csvData)
                resultDict = df.to_dict(orient='records')
                
                del resultDict[0]['Serial_No']
                del resultDict[0]['Input_Text']
                
                logger.info(f"Successfully processed CSV response with {len(resultDict)} records")
                return resultDict[0]
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error occurred while querying SEE endpoint: {str(e)}")
            raise
        finally:
            files['file'][1].close()
            os.unlink(tempCsvPath)

class StudyTitle:
    
    def __init__(self, studyTitle: str, llm: LLM):
        self.studyTitle = studyTitle
        self.llm = llm
        logger.info(f"Starting study title extraction for: {studyTitle}")

    def classifyDisease(self) -> str:
        """
            Classifies the disease from the study title using a language model.
        
            This method retrieves a list of unique diseases from the database, formats a prompt for the language model, and then runs the classification on the study title. If a disease is successfully classified, it is returned. Otherwise, an empty string is returned.
        """     
        
        logger.info("Classifying Disease from Study Title")
        prompt = self.llm.getPrompt("diseaseClassification")
        uniqueDiseases = executeQuery("SELECT distinct disease FROM clinicalstudy.conditions;")
        diseaseList = '[' + '|'.join(item['disease'] for item in uniqueDiseases) + ']'
        
        logger.info("Initializing LLM")
        if not ini.is_chatgpt_initialized():
            ini.init_chatgpt(api_key=self.llm.apiKey)
        
        logger.info("Running LLM classification")
        classifiedDisease = Output.GPT(
            user_prompt=prompt["userPrompt"].format(trialTitle=self.studyTitle, diseaseList=diseaseList),
            system_prompt=prompt["systemPrompt"],
            model=LLM_MODELS.get("openai").get("gpt4_omni"),
            output_option='cont'
        )
        if classifiedDisease != "NaN":
            logger.info(f"Disease classification completed: {classifiedDisease}")
            return classifiedDisease
        return ""
    
    def getDiseaseDetailsPrompt(self, classifiedDisease: str, diseaseDetails: List[Dict[str, str]]) -> Dict[str, Any]:
        """
            Generates a prompt for retrieving disease details based on the classified disease and the available disease details.
            
            Args:
                classifiedDisease (str): The disease name that was classified from the study title.
                diseaseDetails (List[Dict[str, str]]): A list of dictionaries containing the disease category and example details.
            
            Returns:
                Dict[str, Any]: A dictionary containing the formatted prompts and instructions for retrieving the disease details.
        """
        
        logger.info("Generating Disease Details Prompt...")
        diseaseDetailsPrompt = self.llm.getPrompt("diseaseDetails")
        
        categories = [entry['Disease_Category'] for entry in diseaseDetails]
        examples = [entry['Examples'] for entry in diseaseDetails]
        examples = "\n".join([f'[{i+1}] {categories[i]}: {examples[i]}' for i in range(len(categories))])
        categories = ', '.join([f'[{i+1}] {categories[i]}' for i in range(len(categories))])
        
        formattedTargetEntities = diseaseDetailsPrompt["target_entities"].format(diseaseName=classifiedDisease)
        formattedOutputInstructions = diseaseDetailsPrompt["output_instructions"].format(
            diseaseName=classifiedDisease,
            numCategories=len(diseaseDetails),
            categories=categories,
        )
        formattedSystemPrompt = diseaseDetailsPrompt["llm_settings"]["system_prompt"].format(
            diseaseName=classifiedDisease,
            numCategories=len(diseaseDetails),
            categoryExamples=examples
        )
        formattedUserPrompt = diseaseDetailsPrompt["llm_settings"]["user_prompt"].format(
            diseaseName=classifiedDisease,
            numCategories=len(diseaseDetails),
            categoryExamples=examples
        )

        diseaseDetailsPrompt["target_entities"] = formattedTargetEntities
        diseaseDetailsPrompt["output_instructions"] = formattedOutputInstructions
        diseaseDetailsPrompt["llm_settings"]["system_prompt"] = formattedSystemPrompt
        diseaseDetailsPrompt["llm_settings"]["user_prompt"] = formattedUserPrompt
        
        return diseaseDetailsPrompt
    
    def getDiseaseDetails(self, extractedDisease: str) -> Dict[str, Optional[str]]:
        """
            Retrieves the details of the disease classified from the study title.
            
            This method first calls the `classifyDisease` method to determine the disease name from the study title. If a disease is found, it queries the `diseasecategory` table to retrieve the disease category and example details. It then generates a prompt using the `getDiseaseDetailsPrompt` method and calls the `querySEEEndpoint` method to categorize the disease. Finally, it returns a dictionary containing the classified disease name and its category.
            
            If no disease is found in the study title, it returns a dictionary with `Disease` and `Disease_Category` set to `None`.
            
            Args:
                extractedDisease (str): The disease name extracted from the study title.
            
            Returns:
                Dict[str, Optional[str]]: A dictionary containing the classified disease name and its category.
        """      
        
        classifiedDisease = self.classifyDisease()
        
        if classifiedDisease:
            logger.info(f"Disease found in study title: {classifiedDisease}")
            diseaseDetails = executeQuery(f"SELECT Disease_Category, Examples FROM clinicalstudy.diseasecategory WHERE Disease = '{classifiedDisease}';")
            
            diseaseDetailsPrompt = self.getDiseaseDetailsPrompt(classifiedDisease=classifiedDisease, diseaseDetails=diseaseDetails)
            categorisedDisease = self.llm.querySEEEndpoint(disease=extractedDisease, extractionConfig=diseaseDetailsPrompt)

            return {"Disease": classifiedDisease, "Disease_Category": categorisedDisease.get(f"{classifiedDisease}_Category")}
        
        logger.warning("No disease found in study title")
        return {"Disease": None, "Disease_Category": None}

    def extractEntities(self) -> Dict[str, Any]:
        """
            Extracts entities from the study title and retrieves disease details.
            
            This method is responsible for the following tasks:
            1. Extracts entities from the study title using the `querySEEEndpoint` method and the "studyTitleEntityExtraction" prompt.
            2. Renames the "Disease" key in the extracted entities to "PrimaryDisease".
            3. Calls the `getDiseaseDetails` method to get the classified disease and disease category.
            4. Adds the extracted study title entities and the study title itself to the final result dictionary.
            5. Returns the final result dictionary containing the extracted entities and disease details.
        """
        studyTitleEntities = self.llm.querySEEEndpoint(
            studyTitle=self.studyTitle, 
            extractionConfig=self.llm.getPrompt("studyTitleEntityExtraction")
        )
        studyTitleEntities["Primary_Disease"] = studyTitleEntities.pop('Disease')
        finalResult = self.getDiseaseDetails(extractedDisease=studyTitleEntities['Primary_Disease'])
        finalResult['Study_Title_Entities'] = studyTitleEntities
        finalResult["Study_Title"] = self.studyTitle
        
        return finalResult

def executeQuery(query: str) -> Union[List[Dict[str, Any]], Dict[str, str]]:
    """
        Executes a SQL query and returns the results as a list of dictionaries or an error dictionary.
        
        Args:
            query (str): The SQL query to execute.
        
        Returns:
            Union[List[Dict[str, Any]], Dict[str, str]]: The results of the query as a list of dictionaries, or an error dictionary if an exception occurs.
    """
    
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

studyTitle: str =  "Safety and Efficacy of Deep Wave Trabeculoplasty (DWT) in Primary Open Angle Glaucoma and Ocular Hypertension"
llm = LLM(apiKey=os.getenv("OPENAI_API_KEY_MAYANK_AIDWISE_DEMO"))
studyTitleProcessor = StudyTitle(studyTitle=studyTitle, llm=llm)
result = studyTitleProcessor.extractEntities()
