import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from app.config.settings import (
    MODEL_DETECTOR,
    DEVICE,
    CONFIDENCE_THRESHOLD_DETECTOR,
)

class GamblingObjectDetector:
    def __init__(self):
        print(f"Loading object detection model: {MODEL_DETECTOR}")
        self.processor = AutoImageProcessor.from_pretrained(MODEL_DETECTOR)
        self.model = AutoModelForObjectDetection.from_pretrained(MODEL_DETECTOR)
        self.model.to(DEVICE)
        print("Object detector loaded successfully")

    def detect(self, image: Image.Image, threshold: float = CONFIDENCE_THRESHOLD_DETECTOR):
        
        inputs = self.processor(images=image, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]], device=DEVICE)

        results = self.processor.post_process_object_detection(
            outputs=outputs,
            threshold=threshold,          
            target_sizes=target_sizes
        )[0]

        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            detections.append({
                "class": self.model.config.id2label[label.item()],
                "confidence": float(score.item()),
                "bbox": [float(v) for v in box.tolist()] 
            })

        return detections
