#!/usr/bin/env python3
"""
🚀 Aggressive Fish Recognition System with Extreme Augmentation
Target: 100+ embeddings per image with various augmentation strategies
"""

import os
import sys
import tempfile
import logging
from pathlib import Path

# Add advanced fish system to path
sys.path.append('advanced_fish_system')
from advanced_fish_recognition import AdvancedFishSystem

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from scipy.ndimage import gaussian_filter

logging.basicConfig(level=logging.INFO)

class AggressivePreprocessor:
    """🔥 Aggressive Preprocessing with 25+ augmentation strategies"""
    
    def __init__(self):
        self.augmentation_configs = [
            # Original
            {"name": "original", "params": {}},
            
            # Brightness variations (5 levels)
            {"name": "brightness_factor1.4", "params": {"brightness": 1.4}},
            {"name": "brightness_factor1.2", "params": {"brightness": 1.2}},
            {"name": "brightness_factor0.8", "params": {"brightness": 0.8}},
            {"name": "brightness_factor0.6", "params": {"brightness": 0.6}},
            {"name": "brightness_factor0.4", "params": {"brightness": 0.4}},
            
            # Contrast variations (5 levels)
            {"name": "contrast_factor1.5", "params": {"contrast": 1.5}},
            {"name": "contrast_factor1.3", "params": {"contrast": 1.3}},
            {"name": "contrast_factor0.7", "params": {"contrast": 0.7}},
            {"name": "contrast_factor0.5", "params": {"contrast": 0.5}},
            {"name": "contrast_factor0.3", "params": {"contrast": 0.3}},
            
            # Sharpness variations (4 levels)
            {"name": "sharpness_factor2.0", "params": {"sharpness": 2.0}},
            {"name": "sharpness_factor1.5", "params": {"sharpness": 1.5}},
            {"name": "sharpness_factor0.5", "params": {"sharpness": 0.5}},
            {"name": "sharpness_factor0.2", "params": {"sharpness": 0.2}},
            
            # Color variations (4 levels)
            {"name": "color_factor1.4", "params": {"color": 1.4}},
            {"name": "color_factor1.2", "params": {"color": 1.2}},
            {"name": "color_factor0.8", "params": {"color": 0.8}},
            {"name": "color_factor0.6", "params": {"color": 0.6}},
            
            # CLAHE variations (3 levels)
            {"name": "clahe_clip_limit2.0", "params": {"clahe_clip_limit": 2.0}},
            {"name": "clahe_clip_limit4.0", "params": {"clahe_clip_limit": 4.0}},
            {"name": "clahe_clip_limit6.0", "params": {"clahe_clip_limit": 6.0}},
            
            # Gamma corrections (4 levels)
            {"name": "gamma_gamma1.4", "params": {"gamma": 1.4}},
            {"name": "gamma_gamma1.2", "params": {"gamma": 1.2}},
            {"name": "gamma_gamma0.8", "params": {"gamma": 0.8}},
            {"name": "gamma_gamma0.6", "params": {"gamma": 0.6}},
            
            # Histogram equalization
            {"name": "histogram_eq", "params": {"histogram_eq": True}},
            
            # Gaussian blur (2 levels)
            {"name": "gaussian_blur_sigma0.5", "params": {"gaussian_blur": 0.5}},
            {"name": "gaussian_blur_sigma1.0", "params": {"gaussian_blur": 1.0}},
            
            # New extreme augmentations
            {"name": "invert_colors", "params": {"invert": True}},
            {"name": "posterize", "params": {"posterize": 4}},
            {"name": "solarize", "params": {"solarize": 128}},
            {"name": "equalize", "params": {"equalize": True}},
            {"name": "autocontrast", "params": {"autocontrast": True}},
        ]
    
    def extract_fish_region(self, image_path, system):
        """🎯 Extract fish region using detection"""
        try:
            result = system.predict_fish_complete(image_path)
            if result and "bbox" in result:
                # Load original image
                image = cv2.imread(image_path)
                
                # Extract bounding box
                bbox = result["bbox"]
                x1, y1, x2, y2 = map(int, bbox)
                
                # Crop fish region
                fish_region = image[y1:y2, x1:x2]
                
                # Resize to standard size
                fish_region = cv2.resize(fish_region, (224, 224))
                
                return fish_region, result["detection_confidence"]
            
            # Fallback: return resized original
            image = cv2.imread(image_path)
            return cv2.resize(image, (224, 224)), 0.5
            
        except Exception as e:
            logging.warning(f"Fish extraction failed: {e}, using full image")
            image = cv2.imread(image_path)
            return cv2.resize(image, (224, 224)), 0.5
    
    def apply_augmentation(self, image, config):
        """🎨 Apply specific augmentation configuration"""
        # Convert to PIL for easier manipulation
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            image_pil = image
        
        params = config["params"]
        
        # Basic adjustments
        if "brightness" in params:
            enhancer = ImageEnhance.Brightness(image_pil)
            image_pil = enhancer.enhance(params["brightness"])
        
        if "contrast" in params:
            enhancer = ImageEnhance.Contrast(image_pil)
            image_pil = enhancer.enhance(params["contrast"])
        
        if "sharpness" in params:
            enhancer = ImageEnhance.Sharpness(image_pil)
            image_pil = enhancer.enhance(params["sharpness"])
        
        if "color" in params:
            enhancer = ImageEnhance.Color(image_pil)
            image_pil = enhancer.enhance(params["color"])
        
        # Convert to numpy for OpenCV operations
        image_np = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        
        # OpenCV-based augmentations
        if "clahe_clip_limit" in params:
            lab = cv2.cvtColor(image_np, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=params["clahe_clip_limit"])
            l = clahe.apply(l)
            image_np = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        
        if "gamma" in params:
            gamma = params["gamma"]
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            image_np = cv2.LUT(image_np, table)
        
        if "histogram_eq" in params:
            yuv = cv2.cvtColor(image_np, cv2.COLOR_BGR2YUV)
            yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
            image_np = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        
        if "gaussian_blur" in params:
            sigma = params["gaussian_blur"]
            image_np = cv2.GaussianBlur(image_np, (0, 0), sigma)
        
        # Convert back to PIL for special effects
        image_pil = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
        
        # Special PIL operations
        if "invert" in params:
            image_pil = ImageOps.invert(image_pil)
        
        if "posterize" in params:
            image_pil = ImageOps.posterize(image_pil, params["posterize"])
        
        if "solarize" in params:
            image_pil = ImageOps.solarize(image_pil, params["solarize"])
        
        if "equalize" in params:
            image_pil = ImageOps.equalize(image_pil)
        
        if "autocontrast" in params:
            image_pil = ImageOps.autocontrast(image_pil)
        
        # Convert back to BGR numpy array
        final_image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        
        return final_image
    
    def generate_augmented_images(self, fish_region, base_name, temp_dir):
        """🔄 Generate all augmented versions"""
        augmented_paths = []
        
        for i, config in enumerate(self.augmentation_configs):
            try:
                # Apply augmentation
                augmented = self.apply_augmentation(fish_region, config)
                
                # Save augmented image
                aug_filename = f"{base_name}_{i}_{config['name']}.jpg"
                aug_path = os.path.join(temp_dir, aug_filename)
                cv2.imwrite(aug_path, augmented)
                
                augmented_paths.append((aug_path, config['name']))
                
            except Exception as e:
                logging.warning(f"Augmentation {config['name']} failed: {e}")
        
        return augmented_paths


class AggressiveFishRecognitionSystem:
    """🚀 Aggressive Fish Recognition with Extreme Augmentation"""
    
    def __init__(self):
        self.system = AdvancedFishSystem()
        self.preprocessor = AggressivePreprocessor()
        logging.info("🚀 Aggressive Fish Recognition System initialized")
    
    def add_fish_aggressively(self, image_paths, species_name):
        """💪 Add fish with extreme augmentation"""
        total_embeddings = 0
        augmentation_stats = {}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for idx, image_path in enumerate(image_paths, 1):
                logging.info(f"\n📸 Processing image {idx}/{len(image_paths)}: {os.path.basename(image_path)}")
                
                # Extract fish region
                fish_region, conf = self.preprocessor.extract_fish_region(image_path, self.system)
                logging.info(f"   ✅ Fish extracted (conf: {conf:.3f})")
                
                # Generate aggressive augmentations
                base_name = f"Aggressive_{species_name.replace(' ', '_')}_{idx}"
                augmented_paths = self.preprocessor.generate_augmented_images(
                    fish_region, base_name, temp_dir
                )
                
                logging.info(f"   🔄 Generated {len(augmented_paths)} augmented versions")
                
                # Add each augmented image to database
                for aug_path, aug_name in augmented_paths:
                    try:
                        result = self.system.add_fish_to_database_permanent(aug_path, species_name)
                        if result:
                            total_embeddings += 1
                            augmentation_stats[aug_name] = augmentation_stats.get(aug_name, 0) + 1
                            logging.info(f"      ✅ Added {aug_name} embedding")
                        else:
                            logging.warning(f"      ❌ Failed to add {aug_name}")
                    except Exception as e:
                        logging.error(f"      ❌ Error adding {aug_name}: {e}")
        
        return total_embeddings, augmentation_stats
    
    def test_recognition(self, test_images, expected_species):
        """🧪 Test recognition accuracy"""
        correct = 0
        total = len(test_images)
        results = []
        
        logging.info(f"🧪 Testing recognition for '{expected_species}'")
        
        for idx, test_path in enumerate(test_images, 1):
            logging.info(f"\n🔬 Testing {idx}/{total}: {os.path.basename(test_path)}")
            
            try:
                result = self.system.predict_fish_complete(test_path)
                if result:
                    predicted = result.get("predicted_label", "Unknown")
                    confidence = result.get("confidence", 0.0)
                    
                    is_correct = expected_species.lower() in predicted.lower()
                    if is_correct:
                        correct += 1
                        logging.info(f"   ✅ CORRECT: {predicted} (conf: {confidence:.3f})")
                    else:
                        logging.info(f"   ❌ WRONG: {predicted} (conf: {confidence:.3f})")
                    
                    results.append({
                        "image": os.path.basename(test_path),
                        "predicted": predicted,
                        "confidence": confidence,
                        "correct": is_correct
                    })
                else:
                    logging.info(f"   ❌ FAILED: No prediction")
                    results.append({
                        "image": os.path.basename(test_path),
                        "predicted": "FAILED",
                        "confidence": 0.0,
                        "correct": False
                    })
            except Exception as e:
                logging.error(f"   ❌ ERROR: {e}")
                results.append({
                    "image": os.path.basename(test_path),
                    "predicted": f"ERROR: {e}",
                    "confidence": 0.0,
                    "correct": False
                })
        
        accuracy = (correct / total) * 100 if total > 0 else 0
        
        logging.info(f"\n📊 Test Results:")
        logging.info(f"   Accuracy: {accuracy:.1f}% ({correct}/{total})")
        
        return accuracy, results


def main():
    """🚀 Main execution - Aggressive Bandeng Processing"""
    system = AggressiveFishRecognitionSystem()
    
    # Define bandeng images for training
    bandeng_images = [
        "images/bandeng.jpg",
        "images/bandeng1.jpeg", 
        "images/bandeng2.jpg",
        "images/bandeng3.jpg"
    ]
    
    # Filter existing images
    existing_images = [img for img in bandeng_images if os.path.exists(img)]
    
    if not existing_images:
        print("❌ No bandeng images found!")
        return
    
    print(f"🐟 Found {len(existing_images)} bandeng images")
    print(f"🔄 Will generate ~{len(system.preprocessor.augmentation_configs)} augmentations per image")
    print(f"📊 Expected total embeddings: ~{len(existing_images) * len(system.preprocessor.augmentation_configs)}")
    
    # Add fish aggressively 
    total_embeddings, stats = system.add_fish_aggressively(existing_images, "Ikan Bandeng")
    
    print(f"\n📊 Final Results:")
    print(f"   Processed images: {len(existing_images)}/{len(bandeng_images)}")
    print(f"   Total embeddings added: {total_embeddings}")
    print(f"   Augmentations: {stats}")
    print(f"   ✅ Successfully added Ikan Bandeng!")
    print(f"   Total embeddings: {total_embeddings}")
    
    # Test images
    test_images = [
        "images/bandeng4.jpg",
        "images/bandeng5.jpg", 
        "images/bandeng6.jpg"
    ]
    
    existing_test_images = [img for img in test_images if os.path.exists(img)]
    
    if existing_test_images:
        print(f"\n🧪 Testing recognition...")
        print(f"🧩 Test images: {len(existing_test_images)}")
        
        accuracy, results = system.test_recognition(existing_test_images, "Ikan Bandeng")
        
        print(f"\n📊 Final Results:")
        print(f"   Recognition Accuracy: {accuracy:.1f}%")
        print(f"   Total Training Embeddings: {total_embeddings}")
        
        if accuracy < 50:
            print(f"⚠️ Still needs more extreme augmentation or different approach")
        elif accuracy < 80:
            print(f"🔄 Improving, may need additional augmentation")
        else:
            print(f"🎉 Great success with aggressive augmentation!")
        
        print(f"\n🔍 Individual Test Results:")
        for result in results:
            status = "✅" if result["correct"] else "❌"
            print(f"   {status} {result['image']}: {result['predicted']} (conf: {result['confidence']:.3f})")
    else:
        print(f"\n⚠️ No test images found")


if __name__ == "__main__":
    main()