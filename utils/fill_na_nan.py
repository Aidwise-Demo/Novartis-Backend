import pandas as pd
import math


# Function to replace None and NaN values with the string "NA"
def replace_none_nan_with_na(value):
    """
    This function takes a single value as input and replaces it with "NA"
    if the value is either None or a NaN (Not a Number).

    Parameters:
    value: Any type (the input value to check and potentially replace)

    Returns:
    - "NA" if the input value is None or NaN.
    - The original value otherwise.
    """
    # Check if the value is None
    if value is None:
        return "NA"

    # Check if the value is NaN (only applicable for float types)
    # Use math.isnan to identify NaN values
    if isinstance(value, float) and math.isnan(value):
        return "NA"

    # If the value is neither None nor NaN, return it as is
    return value
