import os
from uuid import uuid4
from PIL import Image, ImageDraw, ImageFont
import shutil

RESULT_DIR = "results/inference"
os.makedirs(RESULT_DIR, exist_ok=True)

def save_original_image(image_path):
    """Save original image without any modifications for non_gambling results"""
    filename = f"{uuid4()}.jpg"
    save_path = os.path.join(RESULT_DIR, filename)
    
    shutil.copy(image_path, save_path)
    
    return f"/results/inference/{filename}"

def draw_bboxes(image_path, detections):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        font = ImageFont.load_default()

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cls = det["class"]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1), cls, fill="white", font=font)

    filename = f"{uuid4()}.jpg"
    save_path = os.path.join(RESULT_DIR, filename)
    image.save(save_path)

    return f"/results/inference/{filename}"
