import numpy as np
import easyocr
from PIL import Image
from app.config.settings import TARGET_CLASSES_FOR_OCR

class GamblingOCR:
    def __init__(self):
        print("EasyOCR Ready")

        self.reader = easyocr.Reader(['en'], gpu=True)

    def extract_text(self, image: Image.Image, detections: list):
        ocr_results = []

        for det in detections:
            if det["class"] not in TARGET_CLASSES_FOR_OCR:
                continue

            x1, y1, x2, y2 = map(int, det["bbox"])
            cropped = image.crop((x1, y1, x2, y2))

            result = self.reader.readtext(
                np.array(cropped),
                detail=0,
                paragraph=True
            )

            text = " ".join(result).strip()

            ocr_results.append({
                "class": det["class"],
                "bbox": det["bbox"],
                "ocr_text": text
            })

        return ocr_results
