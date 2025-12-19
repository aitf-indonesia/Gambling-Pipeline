from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from tempfile import NamedTemporaryFile
from contextlib import asynccontextmanager
from uuid import uuid4
from datetime import datetime
from PIL import Image
import numpy as np
import shutil
import os
import json
import logging
import time
import psutil

from app.pipeline.pipeline import GamblingPipeline
from app.utils.metrics import MetricsCollector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Initialize pipeline and metrics
pipeline = GamblingPipeline()
metrics = MetricsCollector()

# Initialize GPU monitoring
gpu_available = False
gpu_handle = None

try:
    import pynvml
    pynvml.nvmlInit()
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    gpu_available = True
except Exception as e:
    logger.warning(f"GPU monitoring not available: {e}")


def get_resource_info():
    info = {}
    
    # CPU
    try:
        info['cpu_percent'] = round(psutil.cpu_percent(interval=0.1), 1)
    except Exception as e:
        info['cpu_percent'] = 0
    
    # GPU
    if gpu_available and gpu_handle:
        try:
            import pynvml
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
            util_info = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
            info['gpu_memory_mb'] = int(mem_info.used / 1024 / 1024)
            info['gpu_utilization'] = util_info.gpu
        except Exception as e:
            info['gpu_memory_mb'] = 0
            info['gpu_utilization'] = 0
    else:
        info['gpu_memory_mb'] = 0
        info['gpu_utilization'] = 0
    
    return info


def warmup_pipeline(pipeline_instance: GamblingPipeline):
    logger.info("[WARMUP] Starting model warmup...")
    warmup_start = time.time()
    
    try:
        t_start = time.time()
        dummy_image = Image.new("RGB", (224, 224), color=(128, 128, 128))
        _ = pipeline_instance.classifier.predict_prob(dummy_image)
        logger.info(f"[WARMUP] ViT Classifier ready ({(time.time() - t_start) * 1000:.0f}ms)")
        
        t_start = time.time()
        dummy_image_det = Image.new("RGB", (640, 640), color=(128, 128, 128))
        _ = pipeline_instance.detector.detect(dummy_image_det)
        logger.info(f"[WARMUP] RT-DETR Detector ready ({(time.time() - t_start) * 1000:.0f}ms)")
        
        t_start = time.time()
        dummy_array = np.zeros((100, 300, 3), dtype=np.uint8)
        dummy_array.fill(200)
        _ = pipeline_instance.ocr.reader.readtext(dummy_array, detail=0)
        logger.info(f"[WARMUP] EasyOCR ready ({(time.time() - t_start) * 1000:.0f}ms)")
        
        total_warmup_time = (time.time() - warmup_start) * 1000
        logger.info(f"[WARMUP] All models warmed up ({total_warmup_time:.0f}ms)")
        
    except Exception as e:
        logger.error(f"[WARMUP] Warmup failed: {str(e)}")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Gambling Detection API...")
    warmup_pipeline(pipeline)
    logger.info("Server is ready to accept requests")
    
    yield
    
    logger.info("Shutting down server...")


app = FastAPI(
    title="Gambling Detection API",
    description="Classification → Object Detection → OCR",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("results/data", exist_ok=True)

app.mount("/results", StaticFiles(directory="results"), name="results")


@app.get("/health")
async def health_check():
    return JSONResponse({
        "status": "ok",
        "timestamp": datetime.now().isoformat()
    })

@app.get("/metrics")
async def get_metrics():
    return JSONResponse(metrics.get_stats())

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        with NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            shutil.copyfileobj(file.file, temp)
            temp_path = temp.name

        result = pipeline.process(temp_path)

        # Get resource info
        resource_info = get_resource_info()

        # Logging performance
        perf = result.get("performance", {})
        logger.info(
            f"[PREDICT] Status: {result['status']} | "
            f"Total: {perf.get('total_ms', 0)}ms | "
            f"Classifier: {perf.get('classifier_ms', 0)}ms | "
            f"OCR: {perf.get('ocr_ms', 0)}ms | "
            f"Detector: {perf.get('detector_ms', 0)}ms | "
            f"Viz: {perf.get('visualization_ms', 0)}ms | "
            f"GPU_Mem: {resource_info['gpu_memory_mb']:.0f}MiB | "
            f"GPU_Util: {resource_info['gpu_utilization']}% | "
            f"CPU: {resource_info['cpu_percent']}%"
        )

        # Record metrics
        metrics.record_request(
            success=True,
            status=result['status'],
            performance=perf
        )

        result_id = str(uuid4())
        result["id"] = result_id
        result["timestamp"] = datetime.now().isoformat()

        json_path = f"results/data/{result_id}.json"
        with open(json_path, "w") as f:
            json.dump(result, f, indent=4)

        return JSONResponse({"success": True, "result": result})

    except Exception as e:
        logger.error(f"[PREDICT] Error: {str(e)}")
        metrics.record_request(success=False, status="error", performance={})
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.get("/results")
async def get_all_results():
    results = []

    for filename in os.listdir("results/data"):
        if filename.endswith(".json"):
            with open(os.path.join("results/data", filename)) as f:
                results.append(json.load(f))
    
    results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

    return JSONResponse({"success": True, "results": results})

@app.get("/result/{result_id}")
async def get_result_detail(result_id: str):
    json_path = f"results/data/{result_id}.json"
    if not os.path.exists(json_path):
        return JSONResponse({"success": False, "message": "Result not found."}, status_code=404)

    with open(json_path) as f:
        data = json.load(f)

    return JSONResponse({"success": True, "result": data})
