# FireRedASR Microservice

A FastAPI-based microservice for speech-to-text transcription using FireRedASR.

## Setup

1. Download the FireRedASR model:
   ```bash
   cd server
   mkdir -p models
   # Download FireRedASR-AED-L & PUNC-BERT model and place it in models/
   git clone https://huggingface.co/FireRedTeam/FireRedChat-punc models/PUNC-BERT
   git clone https://huggingface.co/hfl/chinese-lert-base models/PUNC-BERT/chinese-lert-base
   pushd models/PUNC-BERT && git lfs pull && popd
   git clone https://huggingface.co/FireRedTeam/FireRedASR-AED-L models/FireRedASR-AED-L
   pushd models/FireRedASR-AED-L && git lfs pull && popd
   ```

2. Build the Docker image:
   ```bash
   # 1. Modify Dockerfile based on your CUDA version
   #    for Ampere and newer GPUs
   #       FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
   #    for Volta GPUs (3080, V100)
   #       FROM pytorch/pytorch:2.5.1-cuda11.8-cudnn9-runtime
   # 2. Use tencentyun pypi mirror (in mainland China)
   #       RUN pip install --no-cache-dir -r requirements.txt --index-url http://mirrors.tencentyun.com/pypi/simple/ --trusted-host mirrors.tencentyun.com
   docker build -t fireredasr-service .
   ```

3. Run the container:
   ```bash
   docker run -d \
     -p 8000:8000 \
     --gpus '"cuda:0"' \
     -v $(pwd)/models:/app/models \
     fireredasr-service
   ```

## API Usage

### Transcribe Audio

**Endpoint**: `POST /audio/transcriptions`

**Parameters**:
- `file`: Audio file (multipart/form-data)
- `media_type`: Response format (application/json or text/plain)

**Example using curl**:
```bash
curl -X POST \
  http://localhost:8000/audio/transciptions \
  -H "Content-Type: multipart/form-data" \
  -F file="@audio.wav"
```

**Response**:
```json
{
  "sentences": [{
    "confidence": 0.8,
    "text": "transcribed text here"
    }],
  "wav_file": "audio.wav"
}
```

## Environment Variables

- `FIREREDASR_PATH`: Path to FireRedASR installation (default: /app/fireredasr)
- `MODEL_DIR`: Path to model directory (default: /app/models) 