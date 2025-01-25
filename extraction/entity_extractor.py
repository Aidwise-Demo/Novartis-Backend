from extraction.study_title_processing import StudyTitle  # Import the StudyTitle class for processing study titles
from llm.llm_handler import LLM  # Import the LLM class for interacting with the language model
import os  # For interacting with the operating system (e.g., for reading environment variables)
from dotenv import load_dotenv  # For loading environment variables from a .env file

# Load environment variables from the .env file
load_dotenv()


def entity_extraction(study_title: str):
    """
    This function takes a study title as input, processes it to extract entities, 
    and prints the extracted entities.

    Args:
        study_title (str): The study title to be processed.

    Returns:
        None
    """

    # Initialize the LLM (Language Model) with an API key from environment variables
    llm = LLM(apiKey=os.getenv("OPENAI_API_KEY_MAYANK_AIDWISE_DEMO"))  # Use the OpenAI API key stored in .env file

    # Create an instance of the StudyTitle processor with the provided study title and LLM
    study_title_processor = StudyTitle(studyTitle=study_title, llm=llm)

    # Extract entities from the study title using the extractEntities method of the StudyTitle class
    extracted_entities = study_title_processor.extractEntities()

    return extracted_entities

