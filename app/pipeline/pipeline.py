import time
from PIL import Image
from app.pipeline.classifier import GamblingClassifier
from app.pipeline.detector import GamblingObjectDetector
from app.pipeline.visualizer import draw_bboxes_base64, original_image_to_base64
from app.config.settings import ENABLE_OCR, THRESHOLD_FUSION

if ENABLE_OCR:
    from app.pipeline.ocr import GamblingOCR

class GamblingPipeline:
    def __init__(self):
        self.classifier = GamblingClassifier()
        self.detector = GamblingObjectDetector()
        self.ocr_enabled = ENABLE_OCR
        self.ocr = GamblingOCR() if self.ocr_enabled else None

    def process(self, image_path: str):
        timings = {}
        pipeline_start = time.time()

        # Load image
        t_start = time.time()
        image = Image.open(image_path).convert("RGB")
        timings["image_load_ms"] = round((time.time() - t_start) * 1000, 2)
        
        # 1. ViT Probability
        t_start = time.time()
        prob_vit = self.classifier.predict_prob(image)
        timings["classifier_ms"] = round((time.time() - t_start) * 1000, 2)
        label_vit = "gambling" if prob_vit >= 0.5 else "non_gambling"

        if self.ocr_enabled:
            t_start = time.time()
            prob_ocr, _, _ = self.ocr.classify_gambling_ocr(image)
            timings["ocr_ms"] = round((time.time() - t_start) * 1000, 2)
            prob_fusion = 0.5 * prob_vit + 0.5 * prob_ocr
            label_fusion = "gambling" if prob_fusion >= THRESHOLD_FUSION else "non_gambling"
        else:
            prob_fusion = prob_vit
            label_fusion = label_vit

        if label_fusion == "non_gambling":
            t_start = time.time()
            visualization_path = original_image_to_base64(image_path)
            timings["visualization_ms"] = round((time.time() - t_start) * 1000, 2)
            timings["detector_ms"] = 0
            timings["total_ms"] = round((time.time() - pipeline_start) * 1000, 2)

            return {
                "status": "non_gambling",
                "prob_vit": round(prob_vit, 4),
                "prob_fusion": round(prob_fusion, 4),
                "label_vit": label_vit,
                "label_fusion": label_fusion,
                "detections": [],
                "visualization_path": visualization_path,
                "performance": timings,
            }

        t_start = time.time()
        detections = self.detector.detect(image)
        timings["detector_ms"] = round((time.time() - t_start) * 1000, 2)

        t_start = time.time()
        visualization_path = draw_bboxes_base64(image_path, detections)
        timings["visualization_ms"] = round((time.time() - t_start) * 1000, 2)

        timings["total_ms"] = round((time.time() - pipeline_start) * 1000, 2)

        return {
            "status": "gambling",
            "prob_vit": round(prob_vit, 4),
            "prob_fusion": round(prob_fusion, 4),
            "label_vit": label_vit,
            "label_fusion": label_fusion,
            "detections": detections,
            "visualization_path": visualization_path,
            "performance": timings,
        }
