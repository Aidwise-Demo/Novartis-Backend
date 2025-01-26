# Standard library imports
import os
import time
import logging
import tempfile
import threading
import concurrent.futures
from collections import deque
from functools import wraps
import multiprocessing
from typing import List, Tuple, Dict, Any, Union

# Third-party imports
import pandas as pd
from fastapi.responses import FileResponse
from fastapi.exceptions import HTTPException
from pydantic import BaseModel
from starlette.background import BackgroundTask
from WrappedLLM import Initialize as ini, Output

# Local imports
from API.GPTPrompts import user_prompt, system_prompt
from API.EntityExtractionModels import SingleEntityExtraction as SEE
from API.LoggingSetup import setup_logging
from API.rateLimiter import ThrottleBarrier, CrossProcessesThrottle

setup_logging()
FILENAME: str = os.path.splitext(os.path.basename(__file__))[0]
logger = logging.getLogger(FILENAME)

rateLimit = 40

class Extraction:
    
    @staticmethod
    def extract_entity(throttleBarrier: ThrottleBarrier, input_data: str, target_entity: str, output_instructions: str, model: str = "gpt-4o", response_format: BaseModel = None, settings: dict[str, any] = None, output_option: str = 'cont_cost_prt_min', batchProcess: bool = True) -> Union[str, Tuple[SEE.Output.BatchResponse, float, int]]:
        """
            Extract a target entity from the provided input data using the specified output instructions and model settings.
            
            Args:
                input_data (str): The input data to extract the target entity from.
                target_entity (str): The name of the target entity to extract.
                output_instructions (str): The instructions for formatting the extracted entity.
                model (str, optional): The name of the language model to use for extraction. Defaults to "gpt-4o".
                response_format (BaseModel, optional): The response format to use for the extracted entity.
                settings (dict[str, any], optional): Additional settings for the extraction process, such as max_tokens and temperature.
            
            Returns:
                Union[str, Tuple[SEE.Output.BatchResponse, float, int]]: The extracted entity, formatted according to the output instructions.
        """
                
        try:
            
            EXTRACTED_ENTITY = ""
            MAX_TOKENS = settings.get("max_tokens", 4096)
            TEMPERATURE = settings.get("temperature", 0.1)
            USER_PROMPT = user_prompt.format(target_entity, str(input_data), output_instructions, settings.get("user_prompt", 'No Specific Instructions!'))
            SYSTEM_PROMPT = system_prompt.format(settings.get("system_prompt", 'No Specific Instructions'))
            
            if batchProcess:
                threading.current_thread().name = target_entity
            
            # throttleBarrier.wait()
            
            if ini.is_chatgpt_initialized():
                EXTRACTED_ENTITY = Output.GPT(
                    user_prompt=USER_PROMPT,
                    system_prompt=SYSTEM_PROMPT, 
                    model=model,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    response_format=response_format, 
                    output_option=output_option)
                
                logger.info("Extraction Finished!")
                
                if EXTRACTED_ENTITY.content:
                    return EXTRACTED_ENTITY.content, EXTRACTED_ENTITY.total_cost, EXTRACTED_ENTITY.total_tokens
                
                return EXTRACTED_ENTITY
                
            if ini.is_claude_initialized():
                EXTRACTED_ENTITY = Output.Claude(USER_PROMPT, system_prompt=SYSTEM_PROMPT, model=model,max_tokens=MAX_TOKENS,temperature=TEMPERATURE)
            if ini.is_gemini_initialized():
                EXTRACTED_ENTITY = Output.Gemini(USER_PROMPT, model=model)
            
            logger.info(f"Extracted {target_entity}!")

            return EXTRACTED_ENTITY
        
        except Exception as e:
            logger.exception(f"Error in ExtractEntity: {str(e)}\nContent: {str(input_data)[:5000]}")
            raise HTTPException(status_code=500, detail=str(e))

class ProcessData:
    
    @staticmethod
    def create_batches(df: pd.DataFrame, batch_size: int) -> List[pd.DataFrame]:
        """
        Creates batches with a specified number of items per batch, balancing the
        total text length in the 'Text' column across batches.

        Args:
            df (pd.DataFrame): The input DataFrame to be batched.
            batch_size (int): The number of items per batch.

        Returns:
            List[pd.DataFrame]: A list of DataFrame batches with balanced text length.
        """
        
        # Sort by text length to start balancing
        df = df[['Key', 'Text']].copy()
        df['Text_Length'] = df['Text'].apply(len)
        sorted_df = df.sort_values(by='Text_Length', ascending=False).reset_index(drop=True)
        
        # Distribute rows to balance text length across batches
        batches = [[] for _ in range((len(df) + batch_size - 1) // batch_size)]
        
        # Interleave entries between batches
        for i, row in sorted_df.iterrows():
            batch_index = i % len(batches)
            batches[batch_index].append(row)
        
        # Convert each batch list to a DataFrame
        balanced_batches = [pd.DataFrame(batch).reset_index(drop=True) for batch in batches]
        
        global_index = 1
        # Iterate through each DataFrame and insert the index column
        for df in balanced_batches:
            df.insert(0, 'Index', range(global_index, global_index + len(df)))
            global_index += len(df)  # Update the global index
        
        return balanced_batches
    
    @staticmethod
    def process_input(input_data: list) -> list:
        """
            Processes the input data by extracting entity names and their corresponding entities from the input data.
            
            Args:
                input_data (list): A list containing alternating entity names and their corresponding entities.
            
            Returns:
                list: A list of dictionaries, where each dictionary represents a batch response containing the entity name and its entities.
        """
        output = []
        for entity_name, entities in zip(input_data[::2], input_data[1::2]):
            batch_response = SEE.Output.BatchResponse(
                Entity_Name=entity_name[1],
                Entities=entities[1]
            )
            output.append(SEE.Output.to_dict(batch_response))
        return output
    
    @staticmethod
    def format_entities(input_df: pd.DataFrame, results: List[Tuple[int, List[SEE.Output]]], target_entities: List[str], flatten: bool = True) -> pd.DataFrame:
        """
            Formats the extracted entities from the input data into a pandas DataFrame, maintaining row-wise correspondence with the original input DataFrame.

            Args:
                input_df (pd.DataFrame): The input DataFrame containing the original data.
                results (List[Tuple[int, List[SEE.Output]]]): A list of tuples, where each tuple contains an index and a list of SEE.Output objects.
                target_entities (List[str]): A list of entity names to extract and include in the output DataFrame.
                flatten (bool, optional): If True, flattens the list of extracted entities. If False, keeps the list of extracted entities as a single row. Defaults to True.

            Returns:
                pd.DataFrame: A DataFrame containing the original input data and the extracted entities.
        """
    
        output: List[Any] = []
        result_dictionary: List[Any] = []
        
        results: List[Dict[str, Any]] = ProcessData.process_input(results)
        result_dictionary.extend(result['extracted_entities'] for result in results)

        if flatten:
            for result in result_dictionary:
                output.extend(result)
        else:
            output = result_dictionary[0]

        # Group entities by Serial_No first
        entities_by_serial: Dict[str, Dict[str, Any]] = {}
        for entity in output:
            key = str(entity['Serial_No']).strip()
            if key not in entities_by_serial:
                entities_by_serial[key] = {}
            entities_by_serial[key][entity['Entity_Name']] = entity['Entity_Value']

        # Create result DataFrame
        result_df = pd.DataFrame({
            'Serial_No': input_df['Key'],
            'Input_Text': input_df['Text']
        })

        # Add entity columns maintaining row-wise correspondence
        for entity_name in target_entities:
            entity_values = []
            for idx, key in enumerate(result_df['Serial_No']):
                key_str = str(key).strip()
                value = entities_by_serial.get(key_str, {}).get(entity_name)
                entity_values.append(value)
            result_df[entity_name] = entity_values

        return result_df
        
class ChatGPT:
    
    @staticmethod
    def nameProcess(batch: pd.DataFrame, process: bool = True) -> None:
        """
            Sets the name of the current process or thread based on the batch of data being processed.
            
            Args:
                batch (pd.DataFrame): The batch of data being processed.
                process (bool, optional): If True, sets the name of the current process. If False, sets the name of the current thread. Defaults to True.
            
            Returns:
                None
        """
        min_key = batch['Index'].min()
        max_key = batch['Index'].max()
        if process:
            multiprocessing.current_process().name = f"Batch {min_key}-{max_key}"
        else:
            threading.current_thread().name = f"Batch {min_key}-{max_key} [Length: {len(batch)}]"
            
    @staticmethod
    def create_csv_file_response(data: pd.DataFrame) -> FileResponse:
        """
            Creates a CSV file response from a given DataFrame.
            
            Args:
                data (pd.DataFrame): The DataFrame to be converted to a CSV file.
            
            Returns:
                FileResponse: A FastAPI FileResponse containing the CSV file.
        """
        df = data  # The data is already a DataFrame
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
            df.to_csv(temp_file.name, index=False)
            
        # Return FileResponse
        return FileResponse(
            path=temp_file.name,
            filename="extracted_entities.csv",
            media_type='text/csv',
            background=BackgroundTask(lambda: os.unlink(temp_file.name))
        )
    
    @staticmethod
    def handleBatchThreads(throttleBarrier: ThrottleBarrier, BATCH_DATA: pd.DataFrame, TARGET_ENTITY: str, OUTPUT_FORMAT: str, API_KEY: str, LLM_SETTINGS: Any, MODEL_NAME: str = 'gpt-4o', OUTPUT_OPTION: str = 'cont_cost') -> Tuple[List[SEE.Output.BatchResponse], float, int]:
        """
            Handles the processing of a batch of input data using the ChatGPT language model.
            
            Args:
                INPUT_DATA (pd.DataFrame): The input data to be processed.
                BATCH_DATA (pd.DataFrame): The batch of input data to be processed.
                TARGET_ENTITY (str): The target entity to extract from the input data.
                OUTPUT_FORMAT (str): The output format specification for the target entity.
                API_KEY (str): The API key for the ChatGPT language model.
                LLM_SETTINGS (Any): The settings for the language model.
                MODEL_NAME (str, optional): The name of the language model to use. Defaults to 'gpt-4o'.
                OUTPUT_OPTION (str, optional): The output option to use. Defaults to 'cont_cost'.
            
            Returns:
                Tuple[List[SEE.Output.BatchResponse], float, int]: The extracted entities, the total cost, and the total tokens used.
        """
        try:
            cost: float = 0.0
            tokens: int = 0
            
            ChatGPT.nameProcess(BATCH_DATA, False)
            logger.info(f"Extracting Entity: {TARGET_ENTITY}")
            
            if 'cost' in OUTPUT_OPTION:
                extractedEntities, cost, tokens = Extraction.extract_entity(
                    input_data=BATCH_DATA[['Key','Text']].to_json(orient='records', indent=2),
                    target_entity=TARGET_ENTITY,
                    output_instructions=OUTPUT_FORMAT,
                    model=MODEL_NAME,
                    settings=dict(LLM_SETTINGS),
                    response_format=SEE.Output.BatchResponse,
                    output_option=OUTPUT_OPTION,
                    batchProcess=False,
                    throttleBarrier=throttleBarrier,
                )
            else:
                extractedEntities = Extraction.extract_entity(
                    input_data=BATCH_DATA[['Key','Text']].to_json(orient='records', indent=2),
                    target_entity=TARGET_ENTITY,
                    output_instructions=OUTPUT_FORMAT,
                    model=MODEL_NAME,
                    settings=dict(LLM_SETTINGS),
                    response_format=SEE.Output.BatchResponse,
                    output_option=OUTPUT_OPTION,
                    batchProcess=False,
                    throttleBarrier=throttleBarrier,
                )
                                
            return extractedEntities, cost, tokens
        
        except Exception as e:
            logger.exception(f"Error in handleBatchThreads: {str(e)}")
            return [], 0.0, 0
    
    @staticmethod
    def handleBatchProcess(throttleBarrier: ThrottleBarrier, INPUT_DATA: pd.DataFrame ,BATCH_DATA: pd.DataFrame, TARGET_ENTITIES: List[str], OUTPUT_FORMAT: List[Dict[str, str]], API_KEY: str, LLM_SETTINGS: Any, MODEL_NAME: str = 'gpt-4o', OUTPUT_OPTION: str = 'cont_cost') -> Tuple[pd.DataFrame, float, int]:
        """
            Handles the processing of a batch of data using the ChatGPT model.

            This function is responsible for extracting entities from a batch of input data using the ChatGPT language model. It supports both single-entity and multi-entity extraction, and can handle the cost and token usage tracking for the extraction process.

            Args:
                INPUT_DATA (pd.DataFrame): The input data to be processed.
                BATCH_DATA (pd.DataFrame): The batch of input data to be processed.
                TARGET_ENTITIES (List[str]): The list of target entities to extract from the input data.
                OUTPUT_FORMAT (List[Dict[str, str]]): The output format specification for the target entities.
                API_KEY (str): The API key for the ChatGPT language model.
                LLM_SETTINGS (Any): The settings for the language model.
                MODEL_NAME (str, optional): The name of the language model to use. Defaults to 'gpt-4o'.
                OUTPUT_OPTION (str, optional): The output option to use. Defaults to 'cont_cost'.

            Returns:
                Tuple[pd.DataFrame, float, int]: The extracted entities, the total cost, and the total tokens used.
        """
                
        try:
            
            TOTAL_COST: float = 0.0
            TOTAL_TOKENS: int = 0
            batch_results: List[SEE.Output.BatchResponse] = []                        
            maxWorkers: int = min(len(TARGET_ENTITIES), 8)
            
            setup_logging()
            ChatGPT.nameProcess(BATCH_DATA)
            
            logger.info("Initializing LLM: [ChatGPT]!")
            ini.init_chatgpt(API_KEY)
            logger.info(f"Extracting Entities using Multithreading, {maxWorkers} Entities at a time!")
            
            logger.warning(f"Rate Limit: {rateLimit} requests per second!")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=maxWorkers) as thread_executor:
                thread_futures = []
                for targetEntity in TARGET_ENTITIES:
                    
                    entitySpecificInstructions = next(
                        (item.instructions for item in OUTPUT_FORMAT 
                        if item.target_entity.lower() == targetEntity.lower()),
                        "No Specific Instructions!")
                    
                    thread_futures.append(thread_executor.submit(
                        Extraction.extract_entity,
                        input_data=BATCH_DATA[['Key','Text']].to_json(orient='records', indent=2),
                        target_entity=targetEntity,
                        output_instructions=entitySpecificInstructions,
                        model=MODEL_NAME,
                        response_format=SEE.Output.BatchResponse,
                        settings=dict(LLM_SETTINGS),
                        output_option=OUTPUT_OPTION,
                        throttleBarrier=throttleBarrier
                    ))
                
                logger.info("All Entity Extraction Requests Submitted!")
                
                for future in concurrent.futures.as_completed(thread_futures):
                    if 'cost' in OUTPUT_OPTION:

                        content, cost, tokens = future.result()                        
                        TOTAL_COST += cost
                        TOTAL_TOKENS += tokens
                    else:
                        content = future.result()
                    
                    batch_results.extend(content)

            flattened_results: pd.DataFrame = ProcessData.format_entities(INPUT_DATA, batch_results, TARGET_ENTITIES)
            flattened_results = flattened_results.dropna(subset=flattened_results.columns[2:], how='all')
            
            return flattened_results, TOTAL_COST, TOTAL_TOKENS
    
        except Exception as e:
            logger.exception(f"Error in handleBatch: {str(e)}")
            return pd.DataFrame(), 0.0, 0
    
    @staticmethod
    def handleGPTResponse(*args, **kwargs)-> FileResponse:
        """
        Handles the processing of a batch of data using the ChatGPT model.

        This function is responsible for extracting entities from a batch of input data using the ChatGPT language model. It supports both single-entity and multi-entity extraction, and can handle the cost and token usage tracking for the extraction process.

        Args:
            input_data (pd.DataFrame): The input data to be processed.
            batch_data (pd.DataFrame): The batch of data to be processed.
            target_entities (Union[List, str]): The entity or list of entities to be extracted.
            output_format (str): The format of the output instructions for the extraction process.
            llm_settings (dict): The settings for the language model, including the batch size.
            model_name (str): The name of the language model to be used.
            output_option (str): The option for the output format, including cost and token usage.
            api_key (str): The API key for the ChatGPT model.

        Returns:
            pd.DataFrame: The extracted entities, formatted according to the output instructions.
            float: The total cost of the extraction process.
            int: The total number of tokens used in the extraction process.
        """

        # Extract necessary parameters from kwargs
        INPUT_DATA: pd.DataFrame = kwargs.get('input_data')
        TARGET_ENTITIES: Union[List, str] = kwargs.get('TARGET_ENTITIES')
        OUTPUT_FORMAT: str = kwargs.get('OUTPUT_FORMAT')
        MODEL_NAME: str = kwargs.get('MODEL_NAME', 'gpt-4o')
        LLM_SETTINGS = kwargs.get('LLM_SETTINGS', {})
        OUTPUT_OPTION: str = kwargs.get('OUTPUT_OPTION', 'cont_cost_prt_min')
        TOTAL_COST: float = 0.00
        TOTAL_TOKENS: int = 0
        
        multiprocessing.current_process().name = "MainProcess"
        
        batchSize: int = dict(LLM_SETTINGS).get('batch_size', 20)
        rateLimit = dict(LLM_SETTINGS).get('rate_limit', 40)  # requests per second
        batches: list[pd.DataFrame] = ProcessData.create_batches(INPUT_DATA, batchSize)
        logger.info(f"[{len(batches)}] Batches will Contain {batchSize} Rows Each!")

        crossProcessThrottle = CrossProcessesThrottle(max_requests=rateLimit, per_seconds=1)
        throttleBarrier = crossProcessThrottle.get_barrier()
        
        if isinstance(TARGET_ENTITIES, list):
            
            all_results: pd.DataFrame = pd.DataFrame()
            processCount: int = min(multiprocessing.cpu_count(), len(batches))
            
            logger.info(f"Data will Processed in {processCount} Batches Concurrently!")
            logger.info(f"Extracting {len(TARGET_ENTITIES)} Entities!")            
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=processCount) as process_executor:
                processfutures = {}
                for BATCH in batches:
                    future = process_executor.submit(
                        ChatGPT.handleBatchProcess,
                        INPUT_DATA=INPUT_DATA,
                        BATCH_DATA=BATCH,
                        TARGET_ENTITIES=TARGET_ENTITIES,
                        OUTPUT_FORMAT=OUTPUT_FORMAT,
                        LLM_SETTINGS=LLM_SETTINGS,
                        MODEL_NAME=MODEL_NAME,
                        OUTPUT_OPTION=OUTPUT_OPTION,
                        API_KEY=kwargs.get('API_KEY'),
                        throttleBarrier=throttleBarrier,
                    )
                    processfutures[future] = (BATCH['Index'].min(), BATCH['Index'].max()) 
                
                logger.info(f"All Processes Have been Submitted! Allocating Resources!")
                
                for future in concurrent.futures.as_completed(processfutures):
                    
                    min_idx, max_idx = processfutures[future]
                    # crossProcessThrottle.cycle()
                    
                    try:
                        df, cost, tokens = future.result()
                        
                        TOTAL_COST += cost
                        TOTAL_TOKENS += tokens
                        
                        if TOTAL_COST > 0:
                            logger.info(f"Batch {min_idx} - {max_idx} Cost: ${cost:.6f}")
                            logger.info(f"Batch {min_idx} - {max_idx} Tokens Used: {tokens}")
                        
                        all_results = pd.concat([all_results, df], ignore_index=True)
                        
                    except Exception as e:
                        logger.exception(f"Error in handleBatchProcess: {str(e)}")
                        continue
            all_results = all_results.sort_values('Serial_No').reset_index(drop=True)
            
            if TOTAL_COST > 0:
                logger.warning(f"Complete Total Cost: ${TOTAL_COST:.6f}")
                logger.warning(f"Complete Token Usage: {TOTAL_TOKENS}")
            
            return ChatGPT.create_csv_file_response(all_results)
        
        if not isinstance(TARGET_ENTITIES, list):
            
            logger.warning(f"Rate Limiter Set to {rateLimit} Requests Per Second!")
            
            # Initialize ChatGPT
            logger.info("Initializing LLM: [ChatGPT]!")
            ini.init_chatgpt(kwargs.get('API_KEY'))
            
            logger.info(f"Extracting {TARGET_ENTITIES}!")
            all_results = []
            TOTAL_COST = 0.0
            TOTAL_TOKENS = 0

            max_workers = min(multiprocessing.cpu_count()*4, len(batches))
            # max_workers = 5
            logger.info(f"Processing {len(batches)} batches concurrently using {max_workers} Threads")

            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as thread_executor:
                Threadfutures = {}
                for batch in batches:
                    future = thread_executor.submit(
                        ChatGPT.handleBatchThreads,
                        BATCH_DATA=batch,
                        TARGET_ENTITY=TARGET_ENTITIES,
                        OUTPUT_FORMAT=OUTPUT_FORMAT,
                        API_KEY=kwargs.get('API_KEY'),
                        LLM_SETTINGS=LLM_SETTINGS,
                        MODEL_NAME=MODEL_NAME,
                        OUTPUT_OPTION=OUTPUT_OPTION,
                        throttleBarrier=throttleBarrier,
                    )
                    Threadfutures[future] = (batch['Index'].min(), batch['Index'].max())    

                for future in concurrent.futures.as_completed(Threadfutures):
                    
                    min_idx, max_idx = Threadfutures[future]
                    # crossProcessThrottle.cycle()
                    
                    try:
                        extractedEntities, cost, tokens = future.result()
                        all_results.extend(extractedEntities)
                        
                        TOTAL_COST += cost
                        TOTAL_TOKENS += tokens
                        
                        if TOTAL_COST > 0:
                            logger.info(f"Batch {min_idx} - {max_idx} Cost: ${cost:.6f}")
                            logger.info(f"Batch {min_idx} - {max_idx} Tokens Used: {tokens}")
                            
                    except Exception as e:
                        logger.exception(f"Error in handleBatchThreads: {str(e)}")
                        continue
            
            flattened_results = ProcessData.format_entities(INPUT_DATA, all_results, [TARGET_ENTITIES])
            # flattened_results = flattened_results.dropna(subset=flattened_results.columns[2:], how='all')
            
            logger.info(f"Extraction Job for [{TARGET_ENTITIES}] Finished!")
            
            if TOTAL_COST > 0:
                logger.warning(f"Complete Total Cost: ${TOTAL_COST:.6f}")
                logger.warning(f"Complete Token Usage: {TOTAL_TOKENS}")
            
            return ChatGPT.create_csv_file_response(flattened_results)
                