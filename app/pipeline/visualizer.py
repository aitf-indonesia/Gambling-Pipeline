import os
from uuid import uuid4
from PIL import Image, ImageDraw, ImageFont
import base64
from io import BytesIO

def pil_to_base64(image, format="WEBP"):
    buffered = BytesIO()
    image.save(buffered, format=format, quality=40, method=6)
    buffered.seek(0)

    img_base64 = base64.b64encode(buffered.read()).decode("utf-8")
    mime = format.lower()
    return f"data:image/{mime};base64,{img_base64}"

def original_image_to_base64(image_path):
    image = Image.open(image_path).convert("RGB")
    return pil_to_base64(image)

def draw_bboxes_base64(image_path, detections):
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

    return pil_to_base64(image)