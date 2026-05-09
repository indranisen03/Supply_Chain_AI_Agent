FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Generate synthetic data if Kaggle CSV not present
RUN python data/generate_synthetic_data.py

# Expose ports
EXPOSE 8000 8501

# Default: run FastAPI (override in docker-compose for Streamlit)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
