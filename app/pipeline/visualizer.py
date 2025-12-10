import os
from uuid import uuid4
from PIL import Image, ImageDraw, ImageFont
import shutil

RESULT_DIR = "results/inference"
os.makedirs(RESULT_DIR, exist_ok=True)

def save_original_image(image_path):
    filename = f"{uuid4()}.webp"
    save_path = os.path.join(RESULT_DIR, filename)
    
    image = Image.open(image_path).convert("RGB")
    image.save(save_path, "WebP", quality=85, method=6)
    
    return f"/results/inference/{filename}"

def draw_bboxes(image_path, detections):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        font = ImageFont.load_default()

    class_colors = {
        "menu_nav": "#FF3333",      # Merah
        "logo": "#33FF57",     # Hijau
        "game_thumbnail": "#3357FF",  # Biru
        "cta_button": "#FF9633",  # Oranye
        "banner_promo": "#9B33FF",     # Ungu
    }
    default_color = "#FF3333"  
    
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cls = det["class"]
        
        bbox_color = class_colors.get(cls.lower(), default_color)
        
        draw.rectangle([x1, y1, x2, y2], outline=bbox_color, width=4)
        
        text_bbox = draw.textbbox((x1, y1 - 22), cls, font=font)
        draw.rectangle(text_bbox, fill=bbox_color)
        draw.text((x1, y1 - 22), cls, fill="white", font=font)

    filename = f"{uuid4()}.webp"
    save_path = os.path.join(RESULT_DIR, filename)
    image.save(save_path, "WebP", quality=85, method=6)

    return f"/results/inference/{filename}"
