from datetime import datetime
import mysql.connector
import math

def insert_db(nct_number, study_title, primary_outcome_measures, secondary_outcome_measures,
              inclusion_criteria, exclusion_criteria, response, conn):
    try:
        cursor = conn.cursor()

        # Check if the `history` table exists
        check_table_query = '''
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_name = 'history'
        '''
        cursor.execute(check_table_query)
        table_exists = cursor.fetchone()[0]

        if not table_exists:
            # Create the `history` table if it doesn't exist
            create_table_query = '''
            CREATE TABLE history (
                Serial_Number INT AUTO_INCREMENT PRIMARY KEY,
                NCT_Number VARCHAR(255),
                Study_Title TEXT,
                Primary_Outcome_Measures TEXT,
                Secondary_Outcome_Measures TEXT,
                Inclusion_Criteria TEXT,
                Exclusion_Criteria TEXT,
                Response TEXT,
                timestamp DATETIME NOT NULL
            )
            '''
            cursor.execute(create_table_query)
            print("Table `history` created successfully.")

        # Replace None, empty, 'NA', or NaN values with "Not Available"
        def sanitize_value(value):
            if value is None or value == '' or value == 'NA' or (isinstance(value, float) and math.isnan(value)):
                return 'Not Available'
            return value

        # Sanitize input values
        nct_number = sanitize_value(nct_number)
        study_title = sanitize_value(study_title)
        primary_outcome_measures = sanitize_value(primary_outcome_measures)
        secondary_outcome_measures = sanitize_value(secondary_outcome_measures)
        inclusion_criteria = sanitize_value(inclusion_criteria)
        exclusion_criteria = sanitize_value(exclusion_criteria)
        response = sanitize_value(response)

        # Insert data into the `history` table
        insert_query = '''
            INSERT INTO history (
                NCT_Number, Study_Title, Primary_Outcome_Measures, Secondary_Outcome_Measures,
                Inclusion_Criteria, Exclusion_Criteria, Response, timestamp
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        '''
        current_time = datetime.now()

        # Execute the SQL query with parameters
        cursor.execute(insert_query, (
            nct_number, study_title, primary_outcome_measures, secondary_outcome_measures,
            inclusion_criteria, exclusion_criteria, response, current_time
        ))

        # Commit the transaction
        conn.commit()
        print("Data inserted successfully.")

        cursor.close()
        conn.close()

    except Exception as e:
        print(f"Error inserting data into database: {e}")

# Example usage:
# conn = mysql.connector.connect(host='your_host', user='your_user', password='your_password', database='your_db')
# insert_db("Sample Query", "Sample Response", "NCT123456", "Sample Study Title", "Primary Outcome", "Secondary Outcome", "Inclusion Criteria", "Exclusion Criteria", conn)
