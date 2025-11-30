import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from app.config.settings import MODEL_CLASSIFIER, DEVICE

class GamblingClassifier:
    def __init__(self):
        print(f"Loading classifier model: {MODEL_CLASSIFIER}")
        self.processor = AutoImageProcessor.from_pretrained(MODEL_CLASSIFIER)
        self.model = AutoModelForImageClassification.from_pretrained(MODEL_CLASSIFIER)
        self.model.to(DEVICE)
        print("Classifier loaded successfully")

    def predict(self, image: Image.Image):
        inputs = self.processor(images=image, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        predicted_idx = logits.argmax(-1).item()
        confidence = torch.softmax(logits, dim=-1)[0][predicted_idx].item()
        label = self.model.config.id2label[predicted_idx]

        THRESHOLD = 0.5
        if label == "gambling" and confidence < THRESHOLD:
            label = "non_gambling"

        return label, round(confidence, 4)
