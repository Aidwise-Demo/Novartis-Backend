from typing import List, Union, Optional, Dict
from pydantic import BaseModel, Field, field_validator, Extra
from fastapi.exceptions import HTTPException
from WrappedLLM.LLMModels import LLM_MODELS


class SingleEntityExtraction:
    
    class Input:
        
        class LLMSettings(BaseModel):
            api_key: str
            llm_provider: str
            llm_name: str
            max_tokens: Optional[int] = 1024
            temperature: Optional[float] = 0.5
            user_prompt: Optional[str] = ""
            system_prompt: Optional[str] = "You are a helpful AI assistant."
            batch_size: Optional[int] = 20
            rate_limit: Optional[int] = 40

            @field_validator('llm_name')
            def validate_model_name(cls, value, info):
                model_provider = info.data.get('llm_provider')
                if model_provider and value not in LLM_MODELS.get(model_provider, []):
                    supported_models = LLM_MODELS.get(model_provider, [])
                    raise ValueError({
                        "message": f"Invalid model name '{value}' for the selected provider '{model_provider}'.",
                        "supported_models": list(supported_models.keys())
                    })
                return value
            
            @field_validator('llm_provider')
            def validate_llm_provider(cls, value):
                if value not in LLM_MODELS:
                    raise ValueError('Invalid LLM provider.')
                return value
            
            @field_validator('batch_size')
            def validate_batch_size(cls, value):
                
                if not isinstance(value, int):
                    raise ValueError('Batch size must be an integer')
                
                if not 1 <= value <= 32:
                    raise ValueError('Batch size must be between 1 and 32') 
                return value
            
            @field_validator('rate_limit')
            def validate_rate_limit(cls, value):
                
                if not isinstance(value, int):
                    raise ValueError('Rate Limit must be an integer')
                
                if value > 80:
                    raise ValueError('Rate Limit must be lower than 80!') 
                return value
            
            @field_validator('max_tokens')
            def validate_max_tokens(cls, value):
                if not 16 <= value <= 16384:
                    raise ValueError('max_tokens must be between 16 and 16384')
                return value

            @field_validator('temperature')
            def validate_temperature(cls, value):
                if not 0 <= value <= 1:
                    raise ValueError('temperature must be between 0 and 1')
                return value

            @field_validator('user_prompt', 'system_prompt')
            def validate_prompt_length(cls, value):
                if len(value) > 10000:
                    raise ValueError('Prompt must not exceed 10000 characters')
                return value
        
        
        class InputData(BaseModel):
            key: int
            text: str
            
            @field_validator('text')
            def check_key(cls, value):
                if not isinstance(value, str):
                    raise ValueError('Input must be a string.')
                return value
            @field_validator('key')
            def check_text(cls, value):
                if not isinstance(value, int):
                    raise ValueError('Input must be a Integer.')
                return value

        class ExtractionConfig(BaseModel):
            target_entities: Union[str, List[str]]
            output_instructions: Union[str, List["SingleEntityExtraction.Input.EntityInstructions"]]
            llm_settings: "SingleEntityExtraction.Input.LLMSettings"

            @field_validator('output_instructions')
            def validate_output_instructions(cls, value, values):
                target_entities = values.data.get('target_entities', [])
                
                LIMIT = 15000 
                if isinstance(value, str):
                    
                    if not isinstance(target_entities, str) :
                        raise ValueError('Target entities must be a string when output_instructions is a string')
                    
                    if len(value) > LIMIT:
                        raise ValueError(f'Instructions must not exceed {LIMIT} characters, LESS: {len(value) - LIMIT}')
                
                elif isinstance(value, list):
                    
                    if not isinstance(target_entities, list) :
                        raise ValueError('Target entities must be a list when output_instructions is a list of EntityInstructions')
                    
                    for instruction in value:
                        if instruction.target_entity.lower() not in [entity.lower() for entity in target_entities]:
                            raise ValueError(f"Entity '{instruction.target_entity}' in output_instructions not found in target_entities")
                else:
                    raise ValueError('Output instructions must be either a string or a list of EntityInstructions')
                
                return value

            @field_validator('target_entities')
            def validate_entity(cls, value):
                if isinstance(value, str):
                    if len(value) > 50:
                        raise ValueError('Entity must not exceed 50 characters')
                elif isinstance(value, list):
                    if len(value) > 16:
                        raise ValueError('Entity list must not contain more than 16 items')
                    for item in value:
                        if len(item) > 50:
                            raise ValueError('Each entity in the list must not exceed 50 characters')
                else:
                    raise ValueError('Entity must be a string or a list of strings')
                return value
             
        
        class EntityInstructions(BaseModel):
            target_entity: str
            instructions: str

            class Config:
                extra = 'forbid'

            @field_validator('target_entity')
            def validate_target_entity(cls, value):
                if len(value) > 50:
                    raise ValueError('Target entity must not exceed 50 characters')
                return value

            @field_validator('instructions')
            def validate_instructions(cls, value):
                LIMIT = 500 
                if len(value) > LIMIT:
                    raise ValueError(f'Instructions must not exceed {LIMIT} characters, LESS: {len(value) - LIMIT}')
                return value

            @classmethod
            def validate_keys(cls, values):
                allowed_keys = {'target_entity', 'instructions'}
                input_keys = set(values.keys())
                if input_keys != allowed_keys:
                    extra_keys = input_keys - allowed_keys
                    missing_keys = allowed_keys - input_keys
                    error_msg = []
                    if extra_keys:
                        error_msg.append(f"Extra keys not allowed: {', '.join(extra_keys)}")
                    if missing_keys:
                        error_msg.append(f"Missing required keys: {', '.join(missing_keys)}")
                    raise ValueError('. '.join(error_msg))
                return values
        
        class AvailableModelsInput(BaseModel):
            llm_provider: str

            @field_validator('llm_provider')
            def validate_llm_provider(cls, value):
                # if not isinstance(value, str):
                #     raise ValueError('LLM provider must be a string.')
                if value not in LLM_MODELS and value not in ["All"]:
                    raise ValueError('Invalid LLM provider.')
                return value


    class Output:
        
        class ExtractedEntity(BaseModel):
            Serial_No: Union[int, str]
            Entity_Value: str

        class BatchResponse(BaseModel):
            Entity_Name: str
            Entities: List["SingleEntityExtraction.Output.ExtractedEntity"]
            
        @staticmethod
        def to_dict(batch_response: BatchResponse) -> dict:
            return {
                        'extracted_entities': 
                            [
                                {
                                    'Serial_No': entity.Serial_No,
                                    'Entity_Name': batch_response.Entity_Name,
                                    'Entity_Value': entity.Entity_Value
                                }
                                for entity in batch_response.Entities
                            ]
                    }
            
        # @staticmethod
        # def to_dict(batch_response: BatchResponse) -> dict:
        #     return {
        #                 'extracted_entities': 
        #                     [
        #                         {
        #                             'Serial_No': entity.Serial_No,
        #                             'Entity_Name': entity.Entity_Name,
        #                             'Entity_Value': entity.Entity_Value
        #                         }
        #                         for entity in batch_response
        #                     ]
        #             }
