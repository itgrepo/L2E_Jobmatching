FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy API server and data
COPY 02_API_SERVER/ ./02_API_SERVER/
COPY DATA_SET_V2/ ./DATA_SET_V2/

WORKDIR /app/02_API_SERVER

# HF Spaces requires port 7860
ENV PORT=7860
ENV L2E_MAX_JOBS=3000
ENV L2E_MAX_USERS=10000

EXPOSE 7860

CMD ["python", "api_server.py"]
