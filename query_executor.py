# Standard library imports
import csv
import json
import logging
import os
from typing import Dict, List, Union, Any, Optional

# Third-party imports
from dotenv import load_dotenv
import mysql.connector
from mysql.connector import Error

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

load_dotenv()
apiKey = os.getenv("OPENAI_API_KEY_MAYANK_AIDWISE_DEMO")
endpointURL = os.getenv("SEE_ENDPOINT_URL_AIDWISE_DEMO")


def executeQuery(query: str) -> Union[List[Dict[str, Any]], Dict[str, str]]:
    """
        Executes a SQL query and returns the results as a list of dictionaries or an error dictionary.

        Args:
            query (str): The SQL query to execute.

        Returns:
            Union[List[Dict[str, Any]], Dict[str, str]]: The results of the query as a list of dictionaries, or an error dictionary if an exception occurs.
    """

    # Log the query being executed for debugging purposes
    logger.info(f"Executing query: {query}")
    try:
        # Establish database connection using environment variables for security
        connection = mysql.connector.connect(
            host=os.getenv('AIVENCLOUD_HOST_AIDWISE_DEMO'),
            user=os.getenv('AIVENCLOUD_USERNAME_AIDWISE_DEMO'),
            password=os.getenv('AIVENCLOUD_PASSWORD_AIDWISE_DEMO'),
            database=os.getenv('AIVENCLOUD_DATABASE_AIDWISE_DEMO'),
            port=os.getenv('AIVENCLOUD_PORT_AIDWISE_DEMO')
        )

        # Create a cursor that returns results as dictionaries for easier data handling
        cursor = connection.cursor(dictionary=True)
        # Execute the provided SQL query
        cursor.execute(query)
        # Fetch all results from the query
        results = cursor.fetchall()

        # Clean up resources by closing cursor and connection
        cursor.close()
        connection.close()

        # Log successful query execution and number of results
        logger.info(f"Query executed successfully, returned {len(results)} results")
        return results

    except Error as e:
        # Log and return any database errors that occur during execution
        logger.error(f"Database error occurred: {str(e)}")
        return {"error": str(e)}

