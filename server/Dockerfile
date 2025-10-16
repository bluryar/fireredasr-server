FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# Install system dependencies
RUN apt update
RUN apt install -y git libsndfile1 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clone FireRedASR repository
RUN git clone https://github.com/FireRedTeam/FireRedASR.git fireredasr

# Copy service code
COPY src/ src/
COPY redpost/ fireredasr/redpost/

# Environment variables
ENV PYTHONPATH=/app
ENV FIREREDASR_PATH=/app/fireredasr
ENV MODEL_DIR=/app/models

# Run the service
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"] 