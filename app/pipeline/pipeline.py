import time
from PIL import Image
from app.pipeline.classifier import GamblingClassifier
from app.pipeline.detector import GamblingObjectDetector
from app.pipeline.ocr import GamblingOCR
from app.pipeline.visualizer import draw_bboxes, save_original_image
from app.config.settings import THRESHOLD_FUSION

class GamblingPipeline:
    def __init__(self):
        self.classifier = GamblingClassifier()
        self.detector = GamblingObjectDetector()
        self.ocr = GamblingOCR()

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
        
        # 2. OCR Heuristic (runs for all images)
        t_start = time.time()
        prob_ocr, label_ocr, ocr_text = self.ocr.classify_gambling_ocr(image_path)
        timings["ocr_ms"] = round((time.time() - t_start) * 1000, 2)
        
        # 3. Fusion
        prob_fusion = 0.5 * prob_vit + 0.5 * prob_ocr
        label_fusion = "gambling" if prob_fusion >= THRESHOLD_FUSION else "non_gambling"
        
        # 4. Decision based on fusion
        if label_fusion == "non_gambling":
            # Non-gambling: return early (no RT-DETR, no ocr_text)
            t_start = time.time()
            visualization_path = save_original_image(image_path)
            timings["visualization_ms"] = round((time.time() - t_start) * 1000, 2)
            timings["detector_ms"] = 0  # Tidak dijalankan
            timings["total_ms"] = round((time.time() - pipeline_start) * 1000, 2)
            
            return {
                "status": "non_gambling",
                "prob_vit": round(prob_vit, 4),
                "prob_ocr": round(prob_ocr, 4),
                "prob_fusion": round(prob_fusion, 4),
                "label_vit": label_vit,
                "label_ocr": label_ocr,
                "label_fusion": label_fusion,
                "detections": [],
                "ocr_text": None,
                "visualization_path": visualization_path,
                "performance": timings,
            }
        
        # 5. Gambling: run RT-DETR (all 5 classes)
        t_start = time.time()
        detections = self.detector.detect(image)
        timings["detector_ms"] = round((time.time() - t_start) * 1000, 2)
        
        t_start = time.time()
        visualization_path = draw_bboxes(image_path, detections)
        timings["visualization_ms"] = round((time.time() - t_start) * 1000, 2)
        
        timings["total_ms"] = round((time.time() - pipeline_start) * 1000, 2)
        
        return {
            "status": "gambling",
            "prob_vit": round(prob_vit, 4),
            "prob_ocr": round(prob_ocr, 4),
            "prob_fusion": round(prob_fusion, 4),
            "label_vit": label_vit,
            "label_ocr": label_ocr,
            "label_fusion": label_fusion,
            "detections": detections,
            "ocr_text": ocr_text,
            "visualization_path": visualization_path,
            "performance": timings,
        }
