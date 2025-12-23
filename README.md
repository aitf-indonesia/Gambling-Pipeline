# Gambling Detection Pipeline

An API system for detecting gambling content in images using a multi-stage pipeline combining Vision Transformer (ViT), RT-DETR object detection, and OCR.

## Overview

This system implements a three-stage pipeline for gambling content detection:

1. **ViT Classifier** - Fine-tuned Vision Transformer for binary classification (gambling/non-gambling)
2. **OCR-based Scorer** - Text extraction and keyword-based gambling scoring
3. **RT-DETR Detector** - Real-Time Detection Transformer for object detection (triggered only for gambling content)

The pipeline uses a fusion approach combining ViT and OCR scores to make the final classification decision. If classified as non-gambling, the detector is skipped for performance optimization.

## Architecture

```
Input Image
    ↓
┌─────────────────────────────┐
│  ViT Classifier             │ → prob_vit
│  (aitfindonesia/vit-...)    │
└─────────────────────────────┘
    ↓
┌─────────────────────────────┐
│  OCR-based Scorer           │ → prob_ocr, ocr_text
│  (EasyOCR + Keywords)       │
└─────────────────────────────┘
    ↓
┌─────────────────────────────┐
│  Fusion Decision            │ → prob_fusion
│  (0.5 * vit + 0.5 * ocr)    │
└─────────────────────────────┘
    ↓
    ├─→ Non-Gambling → Return (skip detector)
    │
    └─→ Gambling
            ↓
    ┌─────────────────────────────┐
    │  RT-DETR Detector           │ → bboxes + classes
    │  (aitfindonesia/rtdetr-...) │
    └─────────────────────────────┘
```

## Features

- **Multi-stage Pipeline**: ViT classification, OCR-based scoring, and object detection
- **Performance Optimization**: Conditional execution (detector runs only for gambling content)
- **RESTful API**: FastAPI-based endpoints with CORS support
- **Real-time Monitoring**: GPU/CPU usage tracking and performance metrics
- **Visualization**: Automatic bounding box visualization for detections
- **Result Storage**: JSON-based result persistence with unique IDs
- **Model Warmup**: Pre-loads models on startup for consistent inference speed

## Installation

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support (optional but recommended)
- CUDA runtime provided via system installation or container image

### Setup

> **⚠️ IMPORTANT NOTE**: This setup section is **ONLY** needed if you're setting up on a **CLEAN SERVER from scratch**. 
> 
> If you're using a server with **persistent disk** where all dependencies are already installed, you can **SKIP this entire Setup section** and go directly to the [Deployment](#deployment) section below to run the service.

1. Clone the repository:
```bash
mkdir -p /home/ubuntu/tim5_prd_workdir
cd /home/ubuntu/tim5_prd_workdir
git clone https://github.com/aitf-indonesia/Gambling-Pipeline.git
cd "Gambling-Pipeline"
```

2. Install Miniconda:
```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```

   After installing, close and reopen your terminal application or refresh it by running the following command:
```bash
source ~/miniconda3/bin/activate
```

   Then, initialize conda on all available shells by running the following command:
```bash
conda init --all
```

3. Setup Conda environment:
```bash
conda create -n prd5 python=3.10
```
```bash
conda activate prd5
```

4. Install dependencies:
```bash
pip install --index-url https://download.pytorch.org/whl/cu126 \
  torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1+cu126
```

```bash
pip install -r requirements.txt
```

5. Configure settings in `app/config/settings.py` if needed:
   - Model paths
   - Confidence thresholds
   - Device selection (cuda/cpu)
   - OCR keywords and weights

6. **For deployment on server**, see the [Deployment](#deployment) section

## Usage

### Running the Server

Start the FastAPI server:

```bash
python run_server.py
```

The server will start on `http://127.0.0.1:9090`

### API Endpoints

#### 1. Health Check
```http
GET /health
```

Returns server status and timestamp.

#### 2. Predict
```http
POST /predict
Content-Type: multipart/form-data

file: <image_file>
```

**Response:**
```json
{
  "success": true,
  "result": {
    "id": "uuid",
    "timestamp": "2025-12-21T10:30:00",
    "status": "gambling",
    "prob_vit": 0.8523,
    "prob_ocr": 0.7234,
    "prob_fusion": 0.7878,
    "label_vit": "gambling",
    "label_ocr": "gambling",
    "label_fusion": "gambling",
    "detections": [
      {
        "class": "slot_machine",
        "confidence": 0.92,
        "bbox": [x1, y1, x2, y2]
      }
    ],
    "ocr_text": "extracted text",
    "visualization_path": "data:image/jpeg;base64,...",
    "performance": {
      "image_load_ms": 12.5,
      "classifier_ms": 45.3,
      "ocr_ms": 230.1,
      "detector_ms": 180.7,
      "visualization_ms": 25.4,
      "total_ms": 493.0
    }
  }
}
```

#### 3. Get All Results
```http
GET /results
```

Returns all stored prediction results sorted by timestamp (descending).

#### 4. Get Result by ID
```http
GET /result/{result_id}
```

Retrieves a specific result by its UUID.

#### 5. Get Metrics
```http
GET /metrics
```

Returns server performance statistics and request metrics.

## Models

The pipeline uses fine-tuned models from Hugging Face:

1. **ViT Classifier**: `aitfindonesia/vit-gambling-finetune`
2. **RT-DETR Detector**: `aitfindonesia/rtdetr-r50-gambling-finetune`
3. **OCR**: EasyOCR

## Configuration

### Key Settings (`app/config/settings.py`)

```python
MODEL_CLASSIFIER = "aitfindonesia/komdigiub-gambling-classifier-vit"
MODEL_DETECTOR = "aitfindonesia/komdigiub-gambling-detection-rtdetr-r50"
CONFIDENCE_THRESHOLD_DETECTOR = 0.1
THRESHOLD_FUSION = 0.5
DEVICE = "cuda"
```

### OCR-based Scorer

The system uses keyword matching with Levenshtein distance for fuzzy matching. Keywords are categorized into:
- Core gambling terms
- Slot/RTP/maxwin keywords
- Togel (lottery) terms
- Casino games
- Poker/Domino/QQ games
- Sportsbook terms
- Provider/brand names
- Deposit/withdrawal/bonus terms

## Deployment

```bash
cd /home/ubuntu/tim5_prd_workdir/Gambling-Pipeline
chmod +x setup_prd5.sh
./setup_prd5.sh
```

This will:
1. Install Supervisor
2. Create service configuration
3. Start the API as a background service
4. Enable auto-restart on failure

Check service status:
```bash
sudo supervisorctl status prd5_api
```

## Project Structure

```
.
├── app/
│   ├── api/
│   │   └── api.py              # FastAPI endpoints and middleware
│   ├── config/
│   │   └── settings.py         # Configuration and constants
│   ├── pipeline/
│   │   ├── classifier.py       # ViT classifier implementation
│   │   ├── detector.py         # RT-DETR detector implementation
│   │   ├── ocr.py             # OCR-based scorer logic
│   │   ├── pipeline.py        # Main pipeline orchestration
│   │   └── visualizer.py      # Bounding box visualization
│   └── utils/
│       └── metrics.py         # Performance metrics collector
├── results/
│   └── data/                  # Stored prediction results (JSON)
├── requirements.txt          # Python dependencies
├── run_server.py            # Server entry point
├── setup_prd5.sh           # Deployment script
└── README.md              # This file
```

## Dependencies

See `requirements.txt` for the complete list of **runtime dependencies**.
CUDA and GPU-specific libraries are provided by the execution environment.
