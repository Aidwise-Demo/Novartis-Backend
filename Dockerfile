# Use the official Python 3.12 image
FROM python:3.12-slim
 
# Set the working directory in the container
WORKDIR /app
 
# Copy the requirements.txt into the container
COPY requirements.txt .
 
# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt
 
# Expose port 8000 to the outside world
EXPOSE 8000
 
# Copy the rest of the application code into the container
COPY . .
 
# Command to run the FastAPI application with Uvicorn and Gunicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]