#!/usr/bin/env python3
"""
Enhanced Fish Recognition System dengan preprocessing yang diperkaya
menggunakan deteksi, segmentasi, dan augmentasi untuk embedding yang lebih robust.
"""

import os
import sys
import cv2
import torch
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image, ImageEnhance, ImageFilter
import uuid
from datetime import datetime
import json

# Import model components
sys.path.append('..')
from fish_system import FishRecognitionSystem
from advanced_fish_system.advanced_fish_recognition import AdvancedFishSystem

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedPreprocessor:
    """Enhanced preprocessing with detection, segmentation, and augmentation"""
    
    def __init__(self):
        self.augmentation_strategies = [
            'original',
            'brightness_enhance',
            'contrast_enhance', 
            'sharpness_enhance',
            'color_enhance',
            'gaussian_blur',
            'histogram_equalization',
            'clahe',
            'gamma_correction'
        ]
    
    def apply_brightness_enhancement(self, image: np.ndarray, factor: float = 1.2) -> np.ndarray:
        """Apply brightness enhancement"""
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        enhancer = ImageEnhance.Brightness(pil_image)
        enhanced = enhancer.enhance(factor)
        return cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)
    
    def apply_contrast_enhancement(self, image: np.ndarray, factor: float = 1.3) -> np.ndarray:
        """Apply contrast enhancement"""
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        enhancer = ImageEnhance.Contrast(pil_image)
        enhanced = enhancer.enhance(factor)
        return cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)
    
    def apply_sharpness_enhancement(self, image: np.ndarray, factor: float = 1.5) -> np.ndarray:
        """Apply sharpness enhancement"""
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        enhancer = ImageEnhance.Sharpness(pil_image)
        enhanced = enhancer.enhance(factor)
        return cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)
    
    def apply_color_enhancement(self, image: np.ndarray, factor: float = 1.2) -> np.ndarray:
        """Apply color saturation enhancement"""
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        enhancer = ImageEnhance.Color(pil_image)
        enhanced = enhancer.enhance(factor)
        return cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)
    
    def apply_gaussian_blur(self, image: np.ndarray, sigma: float = 0.5) -> np.ndarray:
        """Apply slight Gaussian blur to reduce noise"""
        return cv2.GaussianBlur(image, (3, 3), sigma)
    
    def apply_histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """Apply histogram equalization"""
        # Convert to YUV and equalize Y channel
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    
    def apply_clahe(self, image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
        """Apply Contrast Limited Adaptive Histogram Equalization"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    def apply_gamma_correction(self, image: np.ndarray, gamma: float = 1.2) -> np.ndarray:
        """Apply gamma correction"""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)
    
    def generate_enhanced_versions(self, image: np.ndarray, num_versions: int = 5) -> List[Tuple[np.ndarray, str]]:
        """Generate multiple enhanced versions of the image"""
        enhanced_versions = []
        
        # Original
        enhanced_versions.append((image.copy(), "original"))
        
        # Generate enhanced versions
        strategies = [
            (self.apply_brightness_enhancement, "brightness_enhanced"),
            (self.apply_contrast_enhancement, "contrast_enhanced"),
            (self.apply_sharpness_enhancement, "sharpness_enhanced"),
            (self.apply_color_enhancement, "color_enhanced"),
            (self.apply_clahe, "clahe_enhanced"),
            (self.apply_gamma_correction, "gamma_corrected"),
        ]
        
        for i, (enhance_func, description) in enumerate(strategies):
            if i >= num_versions - 1:  # -1 because we already have original
                break
            try:
                enhanced = enhance_func(image)
                enhanced_versions.append((enhanced, description))
            except Exception as e:
                logger.warning(f"Failed to apply {description}: {e}")
        
        return enhanced_versions[:num_versions]


class EnhancedFishRecognitionSystem:
    """Enhanced Fish Recognition System dengan preprocessing yang diperkaya"""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.preprocessor = EnhancedPreprocessor()
        self.fish_system = None
        self.advanced_system = None
        self.initialized = False
        
        # Paths
        self.models_dir = self.base_dir / "models"
        self.classification_dir = self.models_dir / "classification"
        self.database_path = self.classification_dir / "database.pt"
        self.labels_path = self.classification_dir / "labels.json"
        
        logger.info("🚀 Enhanced Fish Recognition System initialized")
    
    def initialize(self) -> bool:
        """Initialize the system"""
        try:
            logger.info("🔄 Initializing Enhanced Fish Recognition System...")
            
            # Initialize FishRecognitionSystem untuk basic prediction
            self.fish_system = FishRecognitionSystem()
            logger.info("✅ FishRecognitionSystem loaded")
            
            # Initialize AdvancedFishSystem untuk database management
            self.advanced_system = AdvancedFishSystem()
            logger.info("✅ AdvancedFishSystem loaded")
            
            self.initialized = True
            logger.info("🎉 Enhanced system initialization complete!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize system: {e}")
            return False
    
    def extract_fish_region_with_detection_and_segmentation(self, image_path: str) -> Optional[Dict[str, Any]]:
        """
        Extract fish region using both detection and segmentation for better preprocessing
        """
        try:
            if not os.path.exists(image_path):
                logger.error(f"Image not found: {image_path}")
                return None
            
            # Load image
            img_bgr = cv2.imread(image_path)
            if img_bgr is None:
                logger.error(f"Failed to load image: {image_path}")
                return None
            
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            # 1. Detection untuk mendapatkan bounding box
            detection_results = self.advanced_system.detector.predict(img_rgb)
            
            if not detection_results or len(detection_results) == 0 or len(detection_results[0]) == 0:
                logger.warning("No fish detected")
                return None
            
            best_detection = detection_results[0][0]
            detection_confidence = float(best_detection.score)
            
            # Get bounding box
            bbox = [
                int(best_detection.x1),
                int(best_detection.y1), 
                int(best_detection.x2),
                int(best_detection.y2)
            ]
            
            # Crop detected region
            cropped_bgr = best_detection.get_mask_BGR()
            
            # 2. Segmentation untuk mendapatkan mask yang lebih precise
            segmentation_result = None
            fish_mask = None
            
            try:
                segmentation_result = self.advanced_system.segmentator.predict(img_rgb)
                if segmentation_result is not None:
                    # Create binary mask
                    fish_mask = (segmentation_result > 0.5).astype(np.uint8) * 255
                    
                    # Apply mask to cropped region if possible
                    h, w = cropped_bgr.shape[:2]
                    if fish_mask.shape[:2] == img_rgb.shape[:2]:
                        # Crop mask to match detection region
                        mask_cropped = fish_mask[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                        if mask_cropped.shape[:2] == (h, w):
                            # Apply mask to remove background
                            masked_bgr = cropped_bgr.copy()
                            if len(mask_cropped.shape) == 2:
                                mask_cropped = cv2.cvtColor(mask_cropped, cv2.COLOR_GRAY2BGR)
                            masked_bgr = cv2.bitwise_and(masked_bgr, mask_cropped)
                            cropped_bgr = masked_bgr
                            
            except Exception as e:
                logger.warning(f"Segmentation failed, using detection only: {e}")
            
            return {
                'original_image': img_bgr,
                'cropped_fish': cropped_bgr,
                'bbox': bbox,
                'detection_confidence': detection_confidence,
                'has_segmentation': segmentation_result is not None,
                'fish_mask': fish_mask
            }
            
        except Exception as e:
            logger.error(f"Error extracting fish region: {e}")
            return None
    
    def add_species_with_enhanced_preprocessing(self, image_paths: List[str], species_name: str, 
                                              num_augmentations: int = 5) -> Dict[str, Any]:
        """
        Add new species with enhanced preprocessing and multiple augmentations
        """
        if not self.initialized:
            logger.error("System not initialized")
            return {'success': False, 'error': 'System not initialized'}
        
        logger.info(f"🐟 Adding species '{species_name}' with enhanced preprocessing")
        logger.info(f"📸 Processing {len(image_paths)} images with {num_augmentations} augmentations each")
        
        results = {
            'success': False,
            'species_name': species_name,
            'processed_images': 0,
            'total_embeddings_added': 0,
            'failed_images': [],
            'augmentation_stats': {}
        }
        
        total_embeddings = 0
        processed_count = 0
        
        for i, image_path in enumerate(image_paths, 1):
            logger.info(f"\n📸 Processing image {i}/{len(image_paths)}: {os.path.basename(image_path)}")
            
            try:
                # Extract fish region with detection and segmentation
                fish_data = self.extract_fish_region_with_detection_and_segmentation(image_path)
                
                if fish_data is None:
                    logger.warning(f"❌ Failed to extract fish from {image_path}")
                    results['failed_images'].append(image_path)
                    continue
                
                cropped_fish = fish_data['cropped_fish']
                detection_conf = fish_data['detection_confidence']
                
                logger.info(f"   ✅ Fish extracted (detection conf: {detection_conf:.3f})")
                
                # Generate enhanced versions
                enhanced_versions = self.preprocessor.generate_enhanced_versions(
                    cropped_fish, num_augmentations
                )
                
                logger.info(f"   🔄 Generated {len(enhanced_versions)} enhanced versions")
                
                # Add each enhanced version to database
                for j, (enhanced_image, enhancement_type) in enumerate(enhanced_versions):
                    try:
                        # Save temporary enhanced image
                        temp_filename = f"/tmp/enhanced_{species_name.replace(' ', '_')}_{i}_{j}_{enhancement_type}.jpg"
                        cv2.imwrite(temp_filename, enhanced_image)
                        
                        # Add to database using advanced system
                        success = self.advanced_system.add_fish_to_database_permanent(
                            temp_filename, 
                            species_name
                        )
                        
                        if success:
                            total_embeddings += 1
                            # Track augmentation stats
                            if enhancement_type not in results['augmentation_stats']:
                                results['augmentation_stats'][enhancement_type] = 0
                            results['augmentation_stats'][enhancement_type] += 1
                            
                            logger.info(f"      ✅ Added {enhancement_type} embedding")
                        else:
                            logger.warning(f"      ❌ Failed to add {enhancement_type} embedding")
                        
                        # Clean up temp file
                        if os.path.exists(temp_filename):
                            os.remove(temp_filename)
                            
                    except Exception as e:
                        logger.error(f"      ❌ Error processing {enhancement_type}: {e}")
                
                processed_count += 1
                
            except Exception as e:
                logger.error(f"❌ Error processing {image_path}: {e}")
                results['failed_images'].append(image_path)
        
        results.update({
            'success': processed_count > 0,
            'processed_images': processed_count,
            'total_embeddings_added': total_embeddings,
        })
        
        logger.info(f"\n📊 Enhancement Results:")
        logger.info(f"   Processed images: {processed_count}/{len(image_paths)}")
        logger.info(f"   Total embeddings added: {total_embeddings}")
        logger.info(f"   Augmentation breakdown: {results['augmentation_stats']}")
        
        return results
    
    def test_species_recognition(self, test_images: List[str], expected_species: str) -> Dict[str, Any]:
        """Test species recognition accuracy"""
        if not self.initialized:
            logger.error("System not initialized")
            return {'success': False, 'error': 'System not initialized'}
        
        logger.info(f"🧪 Testing recognition for '{expected_species}'")
        
        results = {
            'expected_species': expected_species,
            'total_tests': len(test_images),
            'correct_predictions': 0,
            'predictions': [],
            'accuracy': 0.0
        }
        
        for i, image_path in enumerate(test_images, 1):
            logger.info(f"\n🔬 Testing {i}/{len(test_images)}: {os.path.basename(image_path)}")
            
            try:
                # Use advanced system for prediction
                prediction_results = self.advanced_system.predict_fish_complete(image_path)
                
                if prediction_results:
                    predicted_species = prediction_results['species']
                    confidence = prediction_results['confidence']
                    
                    is_correct = expected_species.lower() in predicted_species.lower()
                    
                    if is_correct:
                        results['correct_predictions'] += 1
                        logger.info(f"   ✅ CORRECT: {predicted_species} (conf: {confidence:.3f})")
                    else:
                        logger.info(f"   ❌ WRONG: {predicted_species} (conf: {confidence:.3f})")
                    
                    results['predictions'].append({
                        'image': os.path.basename(image_path),
                        'predicted_species': predicted_species,
                        'confidence': confidence,
                        'is_correct': is_correct
                    })
                else:
                    logger.warning(f"   ⚠️ No prediction result")
                    results['predictions'].append({
                        'image': os.path.basename(image_path),
                        'predicted_species': 'No prediction',
                        'confidence': 0.0,
                        'is_correct': False
                    })
                    
            except Exception as e:
                logger.error(f"   ❌ Error testing {image_path}: {e}")
                results['predictions'].append({
                    'image': os.path.basename(image_path),
                    'predicted_species': f'Error: {str(e)}',
                    'confidence': 0.0,
                    'is_correct': False
                })
        
        results['accuracy'] = (results['correct_predictions'] / results['total_tests']) * 100
        
        logger.info(f"\n📊 Test Results:")
        logger.info(f"   Accuracy: {results['accuracy']:.1f}% ({results['correct_predictions']}/{results['total_tests']})")
        
        return results


def main():
    """Main function untuk testing system dengan bandeng"""
    print("🐟 ENHANCED FISH RECOGNITION SYSTEM")
    print("=" * 50)
    
    # Initialize system
    system = EnhancedFishRecognitionSystem()
    
    if not system.initialize():
        print("❌ Failed to initialize system")
        return
    
    # Test dengan ikan bandeng
    bandeng_images = [
        "images/bandeng.jpg",
        "images/bandeng1.jpeg", 
        "images/bandeng2.jpg",
        "images/bandeng3.jpg",
        "images/bandeng4.jpg"
    ]
    
    # Filter hanya gambar yang ada
    existing_bandeng_images = [img for img in bandeng_images if os.path.exists(img)]
    
    print(f"\n📸 Found {len(existing_bandeng_images)} bandeng images")
    
    if len(existing_bandeng_images) == 0:
        print("❌ No bandeng images found")
        return
    
    # Add bandeng with enhanced preprocessing
    print(f"\n🔄 Adding Ikan Bandeng with enhanced preprocessing...")
    add_results = system.add_species_with_enhanced_preprocessing(
        existing_bandeng_images[:3],  # Use first 3 for training
        "Ikan Bandeng",
        num_augmentations=6
    )
    
    if add_results['success']:
        print(f"✅ Successfully added Ikan Bandeng!")
        print(f"   Total embeddings: {add_results['total_embeddings_added']}")
        
        # Test recognition
        if len(existing_bandeng_images) > 3:
            test_images = existing_bandeng_images[3:]  # Use remaining for testing
        else:
            test_images = existing_bandeng_images  # Use all for testing if not enough
        
        print(f"\n🧪 Testing recognition...")
        test_results = system.test_species_recognition(test_images, "Ikan Bandeng")
        
        print(f"\n📊 Final Results:")
        print(f"   Recognition Accuracy: {test_results['accuracy']:.1f}%")
        
        if test_results['accuracy'] >= 50:
            print("🎉 SUCCESS! Enhanced preprocessing improved recognition!")
        else:
            print("⚠️ Need more enhancement or more training data")
    else:
        print("❌ Failed to add species")

if __name__ == "__main__":
    main()