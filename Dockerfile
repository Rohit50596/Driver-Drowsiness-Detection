# Use the official Python image as a base image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir flask opencv-python-headless pygame tensorflow

# Expose the port the app runs on
EXPOSE 6000

ENV FLASK_APP app.py

# Run the application
CMD ["flask", "run", "--host=0.0.0.0", "--port=6000"]
