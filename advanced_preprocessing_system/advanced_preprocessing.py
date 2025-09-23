#!/usr/bin/env python3
"""
🚀 Advanced Fish Preprocessing System
Transforms 1 input image into 10 enhanced images using:
- Fish detection and segmentation  
- Advanced image augmentation
- Quality enhancement techniques
"""

import os
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import albumentations as A
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path for model imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import existing models
try:
    from models.detection.inference import YOLOInference
    from models.segmentation.inference import Inference as SegmentationInference
    from models.classification.inference import EmbeddingClassifier
    EXISTING_MODELS_AVAILABLE = True
    logger.info("✅ Successfully loaded existing models")
except ImportError as e:
    EXISTING_MODELS_AVAILABLE = False
    logger.warning(f"⚠️ Could not import existing models: {e}")

# Fallback YOLO detection
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("⚠️ ultralytics not available as fallback")

class AdvancedFishPreprocessor:
    """🎯 Advanced preprocessing system that converts 1 image to 10 enhanced versions"""
    
    def __init__(self, model_path: str = None, use_full_image: bool = False):
        """Initialize with fish detection and segmentation models"""
        # Model paths
        self.models_base = os.path.join(os.path.dirname(__file__), '..', 'models')
        self.detection_model_path = os.path.join(self.models_base, 'detection', 'model.ts')
        self.segmentation_model_path = os.path.join(self.models_base, 'segmentation', 'model.ts')
        self.classification_model_path = os.path.join(self.models_base, 'classification', 'model.ts')
        self.database_path = os.path.join(self.models_base, 'classification', 'database.pt')
        
        # Configuration
        self.use_full_image = use_full_image  # Option to use full image instead of cropped
        
        # Initialize models
        self.detector = None
        self.segmentator = None
        self.classifier = None
        self.load_models()
        
        # Define 10 preprocessing strategies
        self.preprocessing_strategies = [
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
    
    def load_models(self):
        """Load existing fish detection, segmentation, and classification models"""
        # Load detection model
        if EXISTING_MODELS_AVAILABLE and os.path.exists(self.detection_model_path):
            try:
                self.detector = YOLOInference(
                    model_path=self.detection_model_path,
                    imsz=(640, 640),
                    conf_threshold=0.05,
                    nms_threshold=0.3,
                    yolo_ver='v10'
                )
                logger.info("✅ Detection model loaded successfully")
            except Exception as e:
                logger.warning(f"⚠️ Could not load detection model: {e}")
        
        # Load segmentation model  
        if EXISTING_MODELS_AVAILABLE and os.path.exists(self.segmentation_model_path):
            try:
                self.segmentator = SegmentationInference(
                    model_path=self.segmentation_model_path,
                    image_size=416,
                    threshold=0.5
                )
                logger.info("✅ Segmentation model loaded successfully")
            except Exception as e:
                logger.warning(f"⚠️ Could not load segmentation model: {e}")
        
        # Load classification model (optional for preprocessing)
        if EXISTING_MODELS_AVAILABLE and os.path.exists(self.classification_model_path) and os.path.exists(self.database_path):
            try:
                self.classifier = EmbeddingClassifier(
                    model_path=self.classification_model_path,
                    data_set_path=self.database_path,
                    device='cpu'
                )
                logger.info("✅ Classification model loaded successfully")
            except Exception as e:
                logger.warning(f"⚠️ Could not load classification model: {e}")
        
        # Fallback to YOLO if needed
        if not self.detector and YOLO_AVAILABLE:
            try:
                self.detector = YOLO("yolov8n.pt")
                logger.info("✅ Fallback YOLO detector loaded")
            except Exception as e:
                logger.warning(f"⚠️ Could not load fallback YOLO: {e}")
    
    def detect_fish_region(self, image: np.ndarray) -> Tuple[np.ndarray, float, Dict]:
        """🎯 Detect fish region and return cropped image with metadata"""
        if self.use_full_image or self.detector is None:
            return image, 0.8, {"bbox": [0, 0, image.shape[1], image.shape[0]], "model": "full_image"}
        
        try:
            # Use existing detection model if available
            if isinstance(self.detector, YOLOInference):
                # Preprocess image for our existing model
                images = [image]
                predictions, params = self.detector.preprocess(images)
                
                with torch.no_grad():
                    results = self.detector.model(predictions)
                
                if self.detector.yolo_ver == 'v10':
                    processed_results = self.detector.v10postprocess(results[0])
                else:
                    processed_results = self.detector.v8postprocess(results[0].cpu().numpy().T)
                
                if len(processed_results) > 0:
                    # Get highest confidence detection
                    best_detection = processed_results[0]  # Already sorted by confidence
                    x1, y1, x2, y2, confidence = best_detection[:5]
                    
                    # Convert to int and ensure within image bounds
                    h, w = image.shape[:2]
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    # Add larger padding to avoid cropping (50% instead of 20%)
                    pad_x = int((x2 - x1) * 0.5)
                    pad_y = int((y2 - y1) * 0.5)
                    
                    x1 = max(0, x1 - pad_x)
                    y1 = max(0, y1 - pad_y)
                    x2 = min(w, x2 + pad_x)
                    y2 = min(h, y2 + pad_y)
                    
                    # If the resulting crop is too small relative to original, use full image
                    crop_area = (x2 - x1) * (y2 - y1)
                    original_area = w * h
                    if crop_area < 0.3 * original_area:  # If crop is less than 30% of original
                        logger.info("   📏 Crop too small, using full image")
                        return image, confidence, {
                            "bbox": [0, 0, w, h],
                            "confidence": confidence,
                            "model": "existing_detection_full_fallback"
                        }
                    
                    # Crop fish region
                    fish_region = image[y1:y2, x1:x2]
                    
                    return fish_region, confidence, {
                        "bbox": [x1, y1, x2, y2],
                        "confidence": confidence,
                        "model": "existing_detection"
                    }
            
            # Fallback to ultralytics YOLO if available
            elif YOLO_AVAILABLE and hasattr(self.detector, 'predict'):
                results = self.detector(image)
                
                if len(results) > 0 and len(results[0].boxes) > 0:
                    # Get highest confidence detection
                    boxes = results[0].boxes
                    confidences = boxes.conf.cpu().numpy()
                    best_idx = np.argmax(confidences)
                    
                    bbox = boxes.xyxy[best_idx].cpu().numpy().astype(int)
                    confidence = confidences[best_idx]
                    
                    # Add padding around detection
                    h, w = image.shape[:2]
                    x1, y1, x2, y2 = bbox
                    
                    # Add 20% padding
                    pad_x = int((x2 - x1) * 0.2)
                    pad_y = int((y2 - y1) * 0.2)
                    
                    x1 = max(0, x1 - pad_x)
                    y1 = max(0, y1 - pad_y)
                    x2 = min(w, x2 + pad_x)
                    y2 = min(h, y2 + pad_y)
                    
                    # Crop fish region
                    fish_region = image[y1:y2, x1:x2]
                    
                    return fish_region, confidence, {
                        "bbox": [x1, y1, x2, y2],
                        "confidence": float(confidence),
                        "model": "fallback_yolo"
                    }
            
        except Exception as e:
            logger.warning(f"Detection failed: {e}")
        
        # Return full image if detection fails
        return image, 0.5, {"bbox": [0, 0, image.shape[1], image.shape[0]], "model": "no_detection"}
    
    def resize_with_aspect_ratio(self, image: np.ndarray, target_size: int = 512) -> np.ndarray:
        """Resize image maintaining aspect ratio with padding"""
        h, w = image.shape[:2]
        
        # Calculate scaling factor
        scale = target_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Create square canvas with padding
        canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        canvas.fill(128)  # Gray background
        
        # Center the resized image
        start_y = (target_size - new_h) // 2
        start_x = (target_size - new_w) // 2
        canvas[start_y:start_y + new_h, start_x:start_x + new_w] = resized
        
        return canvas
    
    def center_crop_square(self, image: np.ndarray) -> np.ndarray:
        """Extract center square crop from image"""
        h, w = image.shape[:2]
        size = min(h, w)
        
        start_y = (h - size) // 2
        start_x = (w - size) // 2
        
        return image[start_y:start_y + size, start_x:start_x + size]
    
    def apply_clahe(self, image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    
    def apply_gamma_correction(self, image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        """Apply gamma correction"""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)
    
    def apply_unsharp_mask(self, image: np.ndarray, strength: float = 1.5) -> np.ndarray:
        """Apply unsharp masking for detail enhancement"""
        # Convert to PIL for easier processing
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Apply unsharp mask
        enhanced = pil_image.filter(ImageFilter.UnsharpMask(radius=2, percent=int(strength * 150), threshold=3))
        
        # Convert back to OpenCV format
        return cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)
    
    def apply_preprocessing_strategy(self, image: np.ndarray, strategy: Dict) -> np.ndarray:
        """Apply specific preprocessing strategy"""
        result = image.copy()
        params = strategy["params"]
        
        # Convert to PIL for some operations
        pil_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        
        # Basic enhancements
        if "brightness" in params:
            enhancer = ImageEnhance.Brightness(pil_image)
            pil_image = enhancer.enhance(params["brightness"])
        
        if "contrast" in params:
            enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer.enhance(params["contrast"])
        
        if "sharpness" in params:
            enhancer = ImageEnhance.Sharpness(pil_image)
            pil_image = enhancer.enhance(params["sharpness"])
        
        if "saturation" in params or "color" in params:
            factor = params.get("saturation", params.get("color", 1.0))
            enhancer = ImageEnhance.Color(pil_image)
            pil_image = enhancer.enhance(factor)
        
        # Convert back to OpenCV
        result = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # OpenCV-based operations
        if "clahe" in params:
            result = self.apply_clahe(result, params["clahe"])
        
        if "gamma" in params:
            result = self.apply_gamma_correction(result, params["gamma"])
        
        if "hist_eq" in params and params["hist_eq"]:
            yuv = cv2.cvtColor(result, cv2.COLOR_BGR2YUV)
            yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
            result = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        
        if "gaussian_blur" in params:
            result = cv2.GaussianBlur(result, (0, 0), params["gaussian_blur"])
        
        if "median_filter" in params and params["median_filter"]:
            result = cv2.medianBlur(result, 5)
        
        if "edge_enhance" in params and params["edge_enhance"]:
            # Edge enhancement using kernel
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            result = cv2.filter2D(result, -1, kernel)
        
        if "auto_contrast" in params and params["auto_contrast"]:
            pil_temp = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            pil_temp = ImageOps.autocontrast(pil_temp)
            result = cv2.cvtColor(np.array(pil_temp), cv2.COLOR_RGB2BGR)
        
        if "unsharp_mask" in params and params["unsharp_mask"]:
            result = self.apply_unsharp_mask(result)
        
        if "enhance" in params and params["enhance"]:
            # General enhancement
            result = self.apply_clahe(result, 1.5)
            result = self.apply_gamma_correction(result, 1.1)
        
        return result
    
    def process_single_image(self, image_path: str, output_dir: str = None) -> List[Dict]:
        """🎯 Process single image into 10 enhanced versions"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        logger.info(f"📸 Processing: {os.path.basename(image_path)}")
        
        # Detect and extract fish region
        fish_region, confidence, detection_metadata = self.detect_fish_region(image)
        logger.info(f"   🎯 Fish detected (confidence: {confidence:.3f})")
        
        results = []
        base_name = Path(image_path).stem
        
        # Apply each preprocessing strategy
        for i, strategy in enumerate(self.preprocessing_strategies):
            try:
                # Apply preprocessing
                processed = self.apply_preprocessing_strategy(fish_region, strategy)
                
                # Save processed image if output directory provided
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = os.path.join(output_dir, f"{base_name}_{i+1:02d}_{strategy['name']}.jpg")
                    cv2.imwrite(output_path, processed)
                else:
                    output_path = None
                
                # Store result metadata
                result = {
                    "strategy_id": i + 1,
                    "strategy_name": strategy["name"],
                    "strategy_params": strategy["params"],
                    "output_path": output_path,
                    "processed_image": processed,
                    "detection_metadata": detection_metadata,
                    "original_path": image_path
                }
                
                results.append(result)
                logger.info(f"      ✅ {strategy['name']}")
                
            except Exception as e:
                logger.error(f"      ❌ Failed {strategy['name']}: {e}")
        
        logger.info(f"   📊 Generated {len(results)}/10 processed images")
        return results


class FishDatasetProcessor:
    """📂 Process entire dataset using folder names as labels"""
    
    def __init__(self, preprocessor: AdvancedFishPreprocessor):
        self.preprocessor = preprocessor
    
    def process_dataset(self, dataset_path: str, output_path: str = None) -> Dict:
        """Process dataset with folder structure: dataset_path/species_name/images"""
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise ValueError(f"Dataset path does not exist: {dataset_path}")
        
        results = {
            "dataset_path": str(dataset_path),
            "output_path": output_path,
            "species": {},
            "total_images": 0,
            "total_processed": 0,
            "failed_images": []
        }
        
        # Supported image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # Process each species folder
        for species_folder in dataset_path.iterdir():
            if not species_folder.is_dir():
                continue
            
            species_name = species_folder.name
            logger.info(f"\n🐟 Processing species: {species_name}")
            
            species_results = {
                "species_name": species_name,
                "images": [],
                "image_count": 0,
                "processed_count": 0
            }
            
            # Find all images in species folder
            image_files = []
            for ext in image_extensions:
                image_files.extend(species_folder.glob(f"*{ext}"))
                image_files.extend(species_folder.glob(f"*{ext.upper()}"))
            
            logger.info(f"   Found {len(image_files)} images")
            
            # Process each image
            for image_file in image_files:
                try:
                    # Set output directory for this species
                    if output_path:
                        species_output_dir = os.path.join(output_path, species_name)
                    else:
                        species_output_dir = None
                    
                    # Process image into 10 versions
                    processed_results = self.preprocessor.process_single_image(
                        str(image_file), 
                        species_output_dir
                    )
                    
                    # Store results
                    image_result = {
                        "original_path": str(image_file),
                        "original_name": image_file.name,
                        "processed_versions": processed_results,
                        "versions_count": len(processed_results)
                    }
                    
                    species_results["images"].append(image_result)
                    species_results["processed_count"] += len(processed_results)
                    results["total_processed"] += len(processed_results)
                    
                except Exception as e:
                    logger.error(f"   ❌ Failed to process {image_file.name}: {e}")
                    results["failed_images"].append(str(image_file))
                
                species_results["image_count"] += 1
                results["total_images"] += 1
            
            results["species"][species_name] = species_results
            logger.info(f"   ✅ Processed {species_results['image_count']} images → {species_results['processed_count']} versions")
        
        return results
    
    def save_processing_report(self, results: Dict, report_path: str):
        """Save processing report to JSON file"""
        # Create serializable version (remove numpy arrays and images)
        serializable_results = self._make_serializable(results)
        
        with open(report_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"📄 Processing report saved: {report_path}")
    
    def _make_serializable(self, obj):
        """Remove non-serializable objects for JSON export"""
        if isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                if key not in ['processed_image']:  # Skip image arrays
                    result[key] = self._make_serializable(value)
            return result
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return f"<numpy_array_shape_{obj.shape}>"
        else:
            return obj


def main():
    """🚀 Main function for testing the preprocessing system"""
    # Initialize preprocessor
    preprocessor = AdvancedFishPreprocessor()
    
    # Test single image processing
    print("🧪 Testing single image preprocessing...")
    
    # Example with a single image (you can modify this path)
    test_image = "../images/bandeng.jpg"
    if os.path.exists(test_image):
        results = preprocessor.process_single_image(test_image, "output/single_test")
        print(f"✅ Single image test: {len(results)} versions generated")
    else:
        print(f"⚠️ Test image not found: {test_image}")
    
    # Test dataset processing
    print(f"\n📂 Testing dataset processing...")
    processor = FishDatasetProcessor(preprocessor)
    
    # Example dataset structure:
    # data/
    #   ├── bandeng/
    #   │   ├── bandeng1.jpg
    #   │   └── bandeng2.jpg
    #   └── mujair/
    #       ├── mujair1.jpg
    #       └── mujair2.jpg
    
    dataset_path = "data"
    if os.path.exists(dataset_path):
        results = processor.process_dataset(dataset_path, "output/dataset_processed")
        processor.save_processing_report(results, "output/processing_report.json")
        
        print(f"\n📊 Dataset Processing Summary:")
        print(f"   Total images: {results['total_images']}")
        print(f"   Total processed versions: {results['total_processed']}")
        print(f"   Species found: {len(results['species'])}")
        print(f"   Failed images: {len(results['failed_images'])}")
        
        for species_name, species_data in results['species'].items():
            print(f"   🐟 {species_name}: {species_data['image_count']} images → {species_data['processed_count']} versions")
    else:
        print(f"⚠️ Dataset directory not found: {dataset_path}")
        print(f"Please create the directory structure:")
        print(f"data/")
        print(f"  ├── species1/")
        print(f"  │   ├── image1.jpg")
        print(f"  │   └── image2.jpg")
        print(f"  └── species2/")
        print(f"      ├── image1.jpg")
        print(f"      └── image2.jpg")


if __name__ == "__main__":
    main()