# Use a lightweight official Python image as the base
FROM python:3.9-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container at /app
COPY . .

# Expose the port your Flask application runs on (default Flask port is 5000)
EXPOSE 5000

# Set environment variables for Flask
ENV FLASK_APP=application.py
ENV FLASK_RUN_HOST=0.0.0.0

# Command to run your Flask application when the container starts
CMD ["flask", "run"]