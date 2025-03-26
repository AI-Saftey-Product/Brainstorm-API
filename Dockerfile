FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/var/app/current
ENV PORT=8000
ENV DISABLE_DATABASE=True
ENV DEBUG=False
ENV API_VERSION=v1
ENV API_TITLE="AI Safety Testing API"
ENV API_DESCRIPTION="API for testing AI models for safety concerns"

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]