# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required by OpenCV.
# libgl1-mesa-glx provides the missing libGL.so.1 library.
# libglib2.0-0 provides the missing libgthread-2.0.so.0 library.
# We also clean up the apt cache to keep the image lightweight.
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir ensures that we don't store the pip cache, keeping the image size smaller.
# --upgrade pip ensures we have the latest version of pip.
RUN pip install --no-cache-dir --upgrade pip -r requirements.txt

# Copy the rest of the application's code into the container at /app
# This includes app.py, deepfake_starter_code.py, and the output/ folder with the model.
# The .dockerignore file ensures that large data folders are NOT copied.
COPY . .

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define the command to run the application.
# We use --host 0.0.0.0 to make the app accessible from outside the container.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"] 
