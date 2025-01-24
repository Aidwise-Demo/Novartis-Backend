import csv
import json
import logging
import os
import tempfile
import pandas as pd
import requests
from io import StringIO
from typing import Dict, List, Union, Any, Optional
from WrappedLLM import Output, Initialize as ini
from WrappedLLM.LLMModels import LLM_MODELS

# Logger configuration
logger = logging.getLogger('StudyTitleExtraction')
logger.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter(
    '\033[1m(%(levelname)s)\033[0m \033[1m[%(asctime)s]\033[0m \033[1m[Thread: %(threadName)s]\033[0m [%(funcName)s(Line: %(lineno)d)]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Create file handler
file_handler = logging.FileHandler('study_title_extraction.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

class LLM:
    def __init__(self, apiKey: str):
        self.apiKey = apiKey

    def getPrompt(self, identifier: str) -> Dict[str, Union[str, Any]]:
        """
        Retrieves a dictionary of prompts for various natural language processing tasks, such as disease classification and entity extraction from medical text. The prompts include system and user prompts, output instructions, and LLM settings for the specified task.

            The `get_prompt` method takes an `identifier` parameter, which is a string that specifies the task to retrieve the prompt for. The method returns a dictionary containing the relevant prompts and settings for the specified task.
        """
        # Dictionary containing prompts for different NLP tasks
        prompts = {
            # Configuration for disease classification task
            'diseaseClassification': {
                "systemPrompt": """You are a knowledgeable assistant with access to a wide range of clinical trial information. When given a clinical trial title from the NCT database, classify it into a list of provided diseases based on the content and focus of the trial. Only return the disease title from the provided list, if the trial is classified as a disease. The format of the returned disease must strictly match the one in the list. If the trial cannot be classified, return "NaN". No other text should be included in the response.""",
                "userPrompt": "Given the NCT clinical trial title: \"{trialTitle}\", classify it into the following diseases: {diseaseList}. What disease(s) are associated with this trial?",
            },

            # Configuration for entity extraction from study titles
            'studyTitleEntityExtraction': {
                # Instructions for extracting different types of entities
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
                # List of entities to be extracted
                "target_entities": [
                    "Disease",
                    "Drug",
                    "Trial Phase",
                    "Population Segment"
                ],
                # LLM configuration settings
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

            # Configuration for disease details classification
            'diseaseDetails': {
                "output_instructions": "Classify each input term describing {diseaseName}-related conditions into one of the following {numCategories} categories based on the provided examples and their clinical context: {categories}. Ensure that terms outside the provided examples are classified into the most relevant category based on their description and clinical context. If a term does not match any of the categories, return 'NaN'. If multiple {diseaseName}-related instances are mentioned in a single input, select the most appropriate category based on the clinical context. The output should include the full category name as provided.",
                "target_entities": "{diseaseName}_Category",
                # LLM settings for disease details classification
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

        # Return the prompt configuration for the specified identifier, or an error message if not found
        return prompts.get(identifier, "Prompt not found for the given identifier")

    def querySEEEndpoint(self, extractionConfig: Dict[str, Any], studyTitle: str = None, disease: str = None) -> Dict[
        str, Any]:
        """
            Queries the SEE (Structured Entity Extraction) endpoint with the provided extraction configuration and either a study title or disease name.

            Args:
                extractionConfig (Dict[str, Any]): The extraction configuration to be used for the SEE endpoint.
                studyTitle (str, optional): The study title to extract entities from.
                disease (str, optional): The disease name to categorize.

            Returns:
                Dict[str, Any]: The response from the SEE endpoint, either as a JSON object or a dictionary of extracted entities.
        """

        # Validate input parameters - only one of studyTitle or disease should be provided
        if studyTitle and disease:
            logger.error("Both studyTitle and disease are provided. Please provide only one.")

        # Log the operation being performed based on input
        if studyTitle:
            logger.info(f"Extracting Entities from study title: {studyTitle}")
        if disease:
            logger.info(f"Categorising Disease: {disease}")

        # Create a temporary CSV file to store the input data
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', newline='') as tempCsv:
            writer = csv.writer(tempCsv)
            writer.writerow(['Key', 'Text'])  # Write header row
            writer.writerow([1, studyTitle if studyTitle else disease])  # Write data row
            tempCsvPath = tempCsv.name

        logger.info(f"Created temporary CSV at: {tempCsvPath}")

        # Get the SEE endpoint URL from environment variables
        endpointUrl = os.getenv("SEE_ENDPOINT_URL_AIDWISE_DEMO")

        # Prepare the multipart form data with CSV file and extraction configuration
        files = {
            'file': ('input.csv', open(tempCsvPath, 'rb'), 'text/csv'),
            'ExtractionConfig': (None, json.dumps(extractionConfig), 'application/json')
        }

        try:
            # Send POST request to the SEE endpoint
            response = requests.post(
                url=endpointUrl,
                files=files,
            )

            # Check for HTTP errors in the response
            response.raise_for_status()
            logger.info("Successfully received response from SEE endpoint")

            # Get the content type of the response
            contentType = response.headers.get('Content-Type', '')

            # Handle different response formats (JSON or CSV)
            if 'application/json' in contentType:
                # If response is JSON, it's likely an error message
                result = response.json()
                logger.warning(f"Received JSON response (possible error): {result}")
                return result
            else:
                # If response is CSV, process it into a dictionary
                csvData = StringIO(response.content.decode('utf-8'))
                df = pd.read_csv(csvData)
                resultDict = df.to_dict(orient='records')

                # Remove unnecessary fields from the result
                del resultDict[0]['Serial_No']
                del resultDict[0]['Input_Text']

                logger.info(f"Successfully processed CSV response with {len(resultDict)} records")
                return resultDict[0]

        except requests.exceptions.RequestException as e:
            # Handle any errors that occur during the request
            logger.error(f"Error occurred while querying SEE endpoint: {str(e)}")
            raise
        finally:
            # Clean up resources: close the file and delete the temporary CSV
            files['file'][1].close()
            os.unlink(tempCsvPath)
