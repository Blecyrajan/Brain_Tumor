# Use an official Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Install system packages needed for OpenCV & image processing
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements.txt first for caching
COPY backend/requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the rest of the backend folder
COPY backend/ .

# Expose the port Flask or Gunicorn will use
EXPOSE 8080

# Run the app with Gunicorn
CMD ["gunicorn", "app3:app", "--bind", "0.0.0.0:8080"]
