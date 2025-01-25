import logging
from typing import Dict, List, Any, Optional
from WrappedLLM import Output, Initialize as ini
from WrappedLLM.LLMModels import LLM_MODELS
from llm.llm_handler import LLM
from utils.query_executor import executeQuery

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
file_handler = logging.FileHandler('../study_title_extraction.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

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

        # Log the start of disease classification process
        logger.info("Classifying Disease from Study Title")

        # Get the pre-configured prompt template for disease classification
        prompt = self.llm.getPrompt("diseaseClassification")

        # Query the database to get a list of all unique diseases
        uniqueDiseases = executeQuery("SELECT distinct disease FROM clinicalstudy.conditions;")

        # Format the disease list into a pipe-separated string enclosed in brackets
        diseaseList = '[' + '|'.join(item['disease'] for item in uniqueDiseases) + ']'

        # Initialize ChatGPT if not already initialized
        logger.info("Initializing LLM")
        if not ini.is_chatgpt_initialized():
            ini.init_chatgpt(api_key=self.llm.apiKey)

        # Run the language model to classify the disease from the study title
        logger.info("Running LLM classification")
        classifiedDisease = Output.GPT(
            # Format the user prompt with the study title and list of possible diseases
            user_prompt=prompt["userPrompt"].format(trialTitle=self.studyTitle, diseaseList=diseaseList),
            system_prompt=prompt["systemPrompt"],
            # Use the GPT-4o model configured for omnibus tasks
            model=LLM_MODELS.get("openai").get("gpt4_omni"),
            output_option='cont'
        )

        # If a disease was successfully classified (not "NaN"), return it
        if classifiedDisease != "NaN":
            logger.info(f"Disease classification completed: {classifiedDisease}")
            return classifiedDisease

        # Return empty string if no disease was classified
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

        # Log the start of prompt generation
        logger.info("Generating Disease Details Prompt...")

        # Get the base prompt template for disease details
        diseaseDetailsPrompt = self.llm.getPrompt("diseaseDetails")

        # Extract disease categories and examples from the disease details
        categories = [entry['Disease_Category'] for entry in diseaseDetails]
        examples = [entry['Examples'] for entry in diseaseDetails]

        # Format examples as a numbered list with categories and their corresponding examples
        examples = "\n".join([f'[{i + 1}] {categories[i]}: {examples[i]}' for i in range(len(categories))])

        # Format categories as a comma-separated numbered list
        categories = ', '.join([f'[{i + 1}] {categories[i]}' for i in range(len(categories))])

        # Format the target entities section with the classified disease name
        formattedTargetEntities = diseaseDetailsPrompt["target_entities"].format(diseaseName=classifiedDisease)

        # Format output instructions with disease name, number of categories, and category list
        formattedOutputInstructions = diseaseDetailsPrompt["output_instructions"].format(
            diseaseName=classifiedDisease,
            numCategories=len(diseaseDetails),
            categories=categories,
        )

        # Format the system prompt with disease name, category count, and examples
        formattedSystemPrompt = diseaseDetailsPrompt["llm_settings"]["system_prompt"].format(
            diseaseName=classifiedDisease,
            numCategories=len(diseaseDetails),
            categoryExamples=examples
        )

        # Format the user prompt with disease name, category count, and examples
        formattedUserPrompt = diseaseDetailsPrompt["llm_settings"]["user_prompt"].format(
            diseaseName=classifiedDisease,
            numCategories=len(diseaseDetails),
            categoryExamples=examples
        )

        # Update the prompt template with all formatted sections
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

        # Attempt to classify the disease from the study title
        classifiedDisease = self.classifyDisease()

        # If a disease was successfully classified
        if classifiedDisease:
            # Log the found disease for tracking purposes
            logger.info(f"Disease found in study title: {classifiedDisease}")

            # Query the database to get the disease category and examples for the classified disease
            diseaseDetails = executeQuery(
                f"SELECT Disease_Category, Examples FROM clinicalstudy.diseasecategory WHERE Disease = '{classifiedDisease}';")

            # Generate a prompt for disease details using the classified disease and retrieved details
            diseaseDetailsPrompt = self.getDiseaseDetailsPrompt(classifiedDisease=classifiedDisease,
                                                                diseaseDetails=diseaseDetails)

            # Use the LLM to categorize the extracted disease using the generated prompt
            categorisedDisease = self.llm.querySEEEndpoint(disease=extractedDisease,
                                                           extractionConfig=diseaseDetailsPrompt)

            # Return the classified disease and its category
            return {"Disease": classifiedDisease,
                    "Disease_Category": categorisedDisease.get(f"{classifiedDisease}_Category")}

        # Log a warning if no disease was found in the study title
        logger.warning("No disease found in study title")
        # Return None values if no disease was found
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
        # Extract entities from study title using LLM with predefined extraction configuration
        studyTitleEntities = self.llm.querySEEEndpoint(
            studyTitle=self.studyTitle,
            extractionConfig=self.llm.getPrompt("studyTitleEntityExtraction")
        )

        # Rename 'Disease' key to 'Primary_Disease' for better clarity and consistency
        studyTitleEntities["Primary_Disease"] = studyTitleEntities.pop('Disease')

        # Get detailed disease classification and categorization based on the extracted primary disease
        finalResult = self.getDiseaseDetails(extractedDisease=studyTitleEntities['Primary_Disease'])

        # Add the extracted entities and original study title to the final results
        finalResult['Study_Title_Entities'] = studyTitleEntities
        finalResult["Study_Title"] = self.studyTitle

        # Return the complete dictionary containing all extracted information
        return finalResult

