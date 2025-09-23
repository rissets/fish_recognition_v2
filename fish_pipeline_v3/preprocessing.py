import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import albumentations as A
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 10 preprocessing strategies
PREPROCESSING_STRATEGIES = [
    {"name": "original_enhanced", "params": {"enhance": True}},
    {"name": "bright_contrast", "params": {"brightness": 1.3, "contrast": 1.2}},
    {"name": "sharp_saturation", "params": {"sharpness": 1.5, "saturation": 1.3}},
    {"name": "clahe_gamma", "params": {"clahe": 2.0, "gamma": 1.2}},
    {"name": "histogram_eq", "params": {"hist_eq": True, "brightness": 1.1}},
    {"name": "blur_denoise", "params": {"gaussian_blur": 0.5, "median_filter": True}},
    {"name": "edge_enhance", "params": {"edge_enhance": True, "sharpness": 2.0}},
    {"name": "color_balance", "params": {"auto_contrast": True, "color": 1.4}},
    {"name": "lighting_fix", "params": {"brightness": 0.8, "gamma": 0.9}},
    {"name": "detail_enhance", "params": {"unsharp_mask": True, "contrast": 1.4}}
]

def process_image(image_path, output_dir):
    image = cv2.imread(image_path)
    if image is None:
        logger.warning(f"Image not found: {image_path}")
        return
    base_name = Path(image_path).stem
    os.makedirs(output_dir, exist_ok=True)
    for i, strategy in enumerate(PREPROCESSING_STRATEGIES):
        processed = apply_strategy(image, strategy)
        out_path = os.path.join(output_dir, f"{base_name}_{i+1:02d}_{strategy['name']}.jpg")
        cv2.imwrite(out_path, processed)
        logger.info(f"Saved: {out_path}")

def apply_strategy(image, strategy):
    # ...implementasi sesuai riset sebelumnya...
    img = image.copy()
    # original_enhanced
    if strategy["name"] == "original_enhanced":
        img = cv2.convertScaleAbs(img, alpha=1.1, beta=10)
    # bright_contrast
    elif strategy["name"] == "bright_contrast":
        img = cv2.convertScaleAbs(img, alpha=strategy["params"].get("contrast",1.0), beta=30)
    # sharp_saturation
    elif strategy["name"] == "sharp_saturation":
        pil_img = Image.fromarray(img)
        pil_img = ImageEnhance.Sharpness(pil_img).enhance(strategy["params"].get("sharpness",1.0))
        pil_img = ImageEnhance.Color(pil_img).enhance(strategy["params"].get("saturation",1.0))
        img = np.array(pil_img)
    # ...lanjutkan strategi lain sesuai riset...
    return img

def process_folder(input_folder, output_folder):
    for label in os.listdir(input_folder):
        label_path = os.path.join(input_folder, label)
        if os.path.isdir(label_path):
            out_label_dir = os.path.join(output_folder, label)
            os.makedirs(out_label_dir, exist_ok=True)
            for img_file in os.listdir(label_path):
                img_path = os.path.join(label_path, img_file)
                process_image(img_path, out_label_dir)

if __name__ == "__main__":
    # Contoh penggunaan
    process_folder("dataset", "output/preprocessed")
