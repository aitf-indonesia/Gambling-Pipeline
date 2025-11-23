from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from tempfile import NamedTemporaryFile
from uuid import uuid4
from datetime import datetime
import shutil
import os
import json

from app.pipeline.pipeline import GamblingPipeline


app = FastAPI(
    title="Gambling Detection API",
    description="Classification → Object Detection → OCR",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("results/inference", exist_ok=True)
os.makedirs("results/data", exist_ok=True)

app.mount("/results", StaticFiles(directory="results"), name="results")

pipeline = GamblingPipeline()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        with NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            shutil.copyfileobj(file.file, temp)
            temp_path = temp.name

        result = pipeline.process(temp_path)

        result_id = str(uuid4())
        result["id"] = result_id
        result["timestamp"] = datetime.now().isoformat()

        json_path = f"results/data/{result_id}.json"
        with open(json_path, "w") as f:
            json.dump(result, f, indent=4)

        return JSONResponse({"success": True, "result": result})

    except Exception as e:
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
