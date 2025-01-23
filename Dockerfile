# Use an official Python base image
FROM python:3.10-slim

# Set a working directory in the container
WORKDIR /app

# Copy the requirements.txt file first
COPY requirements.txt /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into /app
COPY . /app

# Expose the Flask port (as per your script it’s 5001)
EXPOSE 5001

# Run your Flask application
CMD ["python", "main.py"]
