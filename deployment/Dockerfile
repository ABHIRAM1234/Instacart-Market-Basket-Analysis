# Dockerfile (Final, Simplified Version)

FROM python:3.9-slim
WORKDIR /app

# Install dependencies
COPY deployment/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application script
COPY deployment/app.py .

# --- CRITICAL CHANGE ---
# Copy the CONTENTS of the artifacts folder directly into the /app directory
COPY artifacts/* .

# Expose port and run the server
EXPOSE 5000
CMD ["gunicorn", "--workers", "1", "--threads", "4", "--bind", "0.0.0.0:5000", "app:app"]