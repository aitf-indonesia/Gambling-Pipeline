from PIL import Image
from app.pipeline.classifier import GamblingClassifier
from app.pipeline.detector import GamblingObjectDetector
from app.pipeline.ocr import GamblingOCR
from app.pipeline.visualizer import draw_bboxes, save_original_image

class GamblingPipeline:
    def __init__(self):
        self.classifier = GamblingClassifier()
        self.detector = GamblingObjectDetector()
        self.ocr = GamblingOCR()

    def process(self, image_path: str):
        image = Image.open(image_path).convert("RGB")

        label, confidence = self.classifier.predict(image)

        if label == "non_gambling":
            visualization_path = save_original_image(image_path)
            return {
                "status": "non_gambling",
                "classification_confidence": confidence,
                "detections": [],
                "ocr": [],
                "visualization_path": visualization_path,
            }

        detections = self.detector.detect(image)
        ocr_results = self.ocr.extract_text(image, detections)
        visualization_path = draw_bboxes(image_path, detections)

        return {
            "status": "gambling",
            "classification_confidence": confidence,
            "detections": detections,
            "ocr": ocr_results,
            "visualization_path": visualization_path,
        }
