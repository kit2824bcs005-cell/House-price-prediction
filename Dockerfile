# Use official lightweight Python image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the backend requirements file and install dependencies
COPY backend/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Ensure Gunicorn is installed
RUN pip install gunicorn

# Copy the entire project context (frontend, backend, data) into the container
COPY . .

# Hugging Face Spaces routes all external internet traffic into Port 7860
EXPOSE 7860

# Launch the Flask application using Gunicorn pointing to port 7860
# 'backend.app:app' routes to the 'app' flask instance inside backend/app.py
CMD ["gunicorn", "-b", "0.0.0.0:7860", "backend.app:app"]
