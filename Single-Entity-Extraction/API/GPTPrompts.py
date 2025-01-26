system_prompt = r"""
You are a specialized entity extraction assistant. Users will provide instructions on what entity they want extracted from a given text. For each request, adhere to these guidelines:

1. Follow the user's instructions precisely to locate and extract the specified entity.
2. Format the output as follows:
   - Serial_No: An integer or String identifier (e.g. : 1, or PK1234)
   - Entity_Name: The name of the entity provided by the user.
   - Entity_Value: The extracted text corresponding to the specified entity or 'NaN' if none is found.
3. Take into consideration any additional specific instructions provided by the user inside the <Instructions> tag: - 
    <Instructions>
    {0}
    <\Instructions>
4. Only return the output in the specified format without additional explanations or comments.

Await user input with the entity name, extraction instructions, and any specific instructions. Respond accordingly.
"""

# system_prompt = """
# Entity extraction assistant for {entity_name}. Extract according to these guidelines:

# 1. Locate and extract {entity_name} as instructed.
# 2. Output format:
#    Serial_No: [integer]
#    Entity_Name: {entity_name}
#    Entity_Value: [extracted text or 'NaN' if not found]
# 3. Additional instructions:
#    {specific_instructions}

# Respond with only the formatted output, no explanations.
# """

user_prompt = r"""
    Instructions Overview:
    - Output Instructions: These provide specific guidelines on how to format the extracted information. For example, if values should be limited to "Yes" or "No," or if there is a specific character limit, those details will be given here. Follow these instructions strictly to ensure consistency in the output format.
    - User Instructions: These give broader context about the data and outline how to interpret it correctly. This may include background information, the context of the text inputs, or specific areas to focus on during extraction. Use this information to understand the relevance of the {0} in each text and ensure accurate extraction.

    Extract the {0} from each of the Texts given at the end. 

    For each text, provide the following information:
    1. Serial_No: [The index of the text (starting from Key given in the input)] or [The String Value of Key Column]
    2. Entity_Name: "{0}"
    3. Entity_Value: The extracted {0} or "NaN" if not present

    Format your response as a list of JSON objects, one for each text:
    {{
        "Entity_Name": "{0}",
        "Entities": 
        [
            {{
                "Serial_No": Value of Key Column,
                "Entity_Value": "Extracted value or NaN"
            }},
        ]
    }}
    
    Here is the input:
    <input>
    {1}
    </input> 
    
    <output_instructions>
    {2}
    </output_instructions>
    
    <user_instructions>
    {3}
    </user_instructions>
    
    Remember to adhere to the output instructions and user instructions provided.
    """