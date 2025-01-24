import pandas as pd
from mysql_connector import get_db_connection

def load_table_from_db(table_name, params=None):
    """
    Load data from a database table based on the provided parameters.
    If no parameters are provided, the entire table is retrieved.

    Args:
        table_name (str): The name of the table to query.
        params (tuple, optional): Parameters to pass into the SQL query for filtering. Default is None.

    Returns:
        pd.DataFrame: A DataFrame containing the data from the specified table.
    """
    conn = get_db_connection()
    try:
        # Build the query
        if params is None:
            # If no parameters, retrieve the whole table
            query = f"SELECT * FROM {table_name}"
        else:
            # If parameters are provided, filter by disease
            query = f"SELECT * FROM {table_name} WHERE LOWER(Disease) = %s"

        # Execute the query and load results into a DataFrame
        df = pd.read_sql(query, conn, params=params)

        return df

    finally:
        # Ensure the connection is closed even if an error occurs
        conn.close()
