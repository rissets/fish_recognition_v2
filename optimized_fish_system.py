#!/usr/bin/env python3
"""
Cleaned and optimized Enhanced Fish Recognition System
dengan preprocessing yang diperkaya dan lebih banyak augmentasi.
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

# Import model components
sys.path.append('.')
from advanced_fish_system.advanced_fish_recognition import AdvancedFishSystem

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimizedPreprocessor:
    """Optimized preprocessing dengan lebih banyak variasi augmentasi"""
    
    def __init__(self):
        # Definisi augmentasi yang lebih komprehensif
        self.augmentation_configs = [
            {'type': 'original', 'params': {}},
            {'type': 'brightness', 'params': {'factor': 1.2}},
            {'type': 'brightness', 'params': {'factor': 0.8}},
            {'type': 'contrast', 'params': {'factor': 1.3}},
            {'type': 'contrast', 'params': {'factor': 0.7}},
            {'type': 'sharpness', 'params': {'factor': 1.5}},
            {'type': 'sharpness', 'params': {'factor': 0.5}},
            {'type': 'color', 'params': {'factor': 1.2}},
            {'type': 'color', 'params': {'factor': 0.8}},
            {'type': 'clahe', 'params': {'clip_limit': 2.0}},
            {'type': 'clahe', 'params': {'clip_limit': 4.0}},
            {'type': 'gamma', 'params': {'gamma': 1.2}},
            {'type': 'gamma', 'params': {'gamma': 0.8}},
            {'type': 'histogram_eq', 'params': {}},
            {'type': 'gaussian_blur', 'params': {'sigma': 0.5}},
        ]
    
    def apply_augmentation(self, image: np.ndarray, aug_config: Dict) -> np.ndarray:
        """Apply specific augmentation based on config"""
        aug_type = aug_config['type']
        params = aug_config['params']
        
        try:
            if aug_type == 'original':
                return image.copy()
            
            elif aug_type == 'brightness':
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                enhancer = ImageEnhance.Brightness(pil_image)
                enhanced = enhancer.enhance(params['factor'])
                return cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)
            
            elif aug_type == 'contrast':
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                enhancer = ImageEnhance.Contrast(pil_image)
                enhanced = enhancer.enhance(params['factor'])
                return cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)
            
            elif aug_type == 'sharpness':
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                enhancer = ImageEnhance.Sharpness(pil_image)
                enhanced = enhancer.enhance(params['factor'])
                return cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)
            
            elif aug_type == 'color':
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                enhancer = ImageEnhance.Color(pil_image)
                enhanced = enhancer.enhance(params['factor'])
                return cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)
            
            elif aug_type == 'clahe':
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                clahe = cv2.createCLAHE(clipLimit=params['clip_limit'], tileGridSize=(8,8))
                lab[:,:,0] = clahe.apply(lab[:,:,0])
                return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            elif aug_type == 'gamma':
                inv_gamma = 1.0 / params['gamma']
                table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
                return cv2.LUT(image, table)
            
            elif aug_type == 'histogram_eq':
                yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
                return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
            
            elif aug_type == 'gaussian_blur':
                return cv2.GaussianBlur(image, (3, 3), params['sigma'])
            
            else:
                logger.warning(f"Unknown augmentation type: {aug_type}")
                return image.copy()
                
        except Exception as e:
            logger.warning(f"Failed to apply {aug_type}: {e}")
            return image.copy()
    
    def generate_augmented_versions(self, image: np.ndarray, max_augmentations: int = 15) -> List[Tuple[np.ndarray, str]]:
        """Generate multiple augmented versions of the image"""
        augmented_versions = []
        
        # Limit augmentations to available configs or max_augmentations
        num_augs = min(len(self.augmentation_configs), max_augmentations)
        
        for i in range(num_augs):
            config = self.augmentation_configs[i]
            augmented_image = self.apply_augmentation(image, config)
            
            # Create descriptive name
            aug_name = config['type']
            if config['params']:
                param_str = '_'.join([f"{k}{v}" for k, v in config['params'].items()])
                aug_name = f"{aug_name}_{param_str}"
            
            augmented_versions.append((augmented_image, aug_name))
        
        return augmented_versions


class OptimizedFishRecognitionSystem:
    """Optimized Fish Recognition System dengan preprocessing yang diperkaya"""
    
    def __init__(self):
        self.preprocessor = OptimizedPreprocessor()
        self.advanced_system = None
        self.initialized = False
        
        logger.info("🚀 Optimized Fish Recognition System initialized")
    
    def initialize(self) -> bool:
        """Initialize the system"""
        try:
            logger.info("🔄 Initializing system...")
            
            # Initialize AdvancedFishSystem
            self.advanced_system = AdvancedFishSystem()
            
            self.initialized = True
            logger.info("🎉 System initialization complete!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize: {e}")
            return False
    
    def extract_and_preprocess_fish(self, image_path: str) -> Optional[Dict[str, Any]]:
        """Extract fish region with detection (simplified - no segmentation for now)"""
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
            
            # Detection
            detection_results = self.advanced_system.detector.predict(img_rgb)
            
            if not detection_results or len(detection_results) == 0 or len(detection_results[0]) == 0:
                logger.warning("No fish detected")
                return None
            
            best_detection = detection_results[0][0]
            detection_confidence = float(best_detection.score)
            
            # Get cropped fish
            cropped_bgr = best_detection.get_mask_BGR()
            
            return {
                'original_image': img_bgr,
                'cropped_fish': cropped_bgr,
                'detection_confidence': detection_confidence
            }
            
        except Exception as e:
            logger.error(f"Error extracting fish: {e}")
            return None
    
    def add_species_with_extensive_augmentation(self, image_paths: List[str], species_name: str) -> Dict[str, Any]:
        """Add species with extensive augmentation"""
        if not self.initialized:
            logger.error("System not initialized")
            return {'success': False, 'error': 'System not initialized'}
        
        logger.info(f"🐟 Adding '{species_name}' with extensive augmentation")
        logger.info(f"📸 Processing {len(image_paths)} images")
        
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
                # Extract fish
                fish_data = self.extract_and_preprocess_fish(image_path)
                
                if fish_data is None:
                    logger.warning(f"❌ Failed to extract fish from {image_path}")
                    results['failed_images'].append(image_path)
                    continue
                
                cropped_fish = fish_data['cropped_fish']
                detection_conf = fish_data['detection_confidence']
                
                logger.info(f"   ✅ Fish extracted (conf: {detection_conf:.3f})")
                
                # Generate augmented versions
                augmented_versions = self.preprocessor.generate_augmented_versions(cropped_fish)
                
                logger.info(f"   🔄 Generated {len(augmented_versions)} augmented versions")
                
                # Add each augmented version to database
                for j, (augmented_image, aug_name) in enumerate(augmented_versions):
                    try:
                        # Save temporary augmented image
                        temp_filename = f"/tmp/optimized_{species_name.replace(' ', '_')}_{i}_{j}_{aug_name}.jpg"
                        cv2.imwrite(temp_filename, augmented_image)
                        
                        # Add to database
                        success = self.advanced_system.add_fish_to_database_permanent(
                            temp_filename, 
                            species_name
                        )
                        
                        if success:
                            total_embeddings += 1
                            # Track augmentation stats
                            if aug_name not in results['augmentation_stats']:
                                results['augmentation_stats'][aug_name] = 0
                            results['augmentation_stats'][aug_name] += 1
                            
                            logger.info(f"      ✅ Added {aug_name} embedding")
                        else:
                            logger.warning(f"      ❌ Failed to add {aug_name} embedding")
                        
                        # Clean up temp file
                        if os.path.exists(temp_filename):
                            os.remove(temp_filename)
                            
                    except Exception as e:
                        logger.error(f"      ❌ Error processing {aug_name}: {e}")
                
                processed_count += 1
                
            except Exception as e:
                logger.error(f"❌ Error processing {image_path}: {e}")
                results['failed_images'].append(image_path)
        
        results.update({
            'success': processed_count > 0,
            'processed_images': processed_count,
            'total_embeddings_added': total_embeddings,
        })
        
        logger.info(f"\n📊 Final Results:")
        logger.info(f"   Processed images: {processed_count}/{len(image_paths)}")
        logger.info(f"   Total embeddings added: {total_embeddings}")
        logger.info(f"   Augmentations: {results['augmentation_stats']}")
        
        return results
    
    def test_recognition_accuracy(self, test_images: List[str], expected_species: str) -> Dict[str, Any]:
        """Test recognition accuracy"""
        if not self.initialized:
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
                prediction = self.advanced_system.predict_fish_complete(image_path)
                
                if prediction:
                    predicted_species = prediction['species']
                    confidence = prediction['confidence']
                    
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
                    logger.warning(f"   ⚠️ No prediction")
                    results['predictions'].append({
                        'image': os.path.basename(image_path),
                        'predicted_species': 'No prediction',
                        'confidence': 0.0,
                        'is_correct': False
                    })
                    
            except Exception as e:
                logger.error(f"   ❌ Error: {e}")
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
    """Main function untuk testing optimized system dengan bandeng"""
    print("🐟 OPTIMIZED FISH RECOGNITION SYSTEM")
    print("=" * 50)
    
    # Initialize system
    system = OptimizedFishRecognitionSystem()
    
    if not system.initialize():
        print("❌ Failed to initialize system")
        return
    
    # Test dengan ikan bandeng
    bandeng_images = [
        "images/bandeng.jpg",
        "images/bandeng1.jpeg", 
        "images/bandeng2.jpg",
        "images/bandeng3.jpg",
        "images/bandeng4.jpg",
        "images/bandeng5.jpg",
        "images/bandeng6.jpg"
    ]
    
    # Filter hanya gambar yang ada
    existing_bandeng_images = [img for img in bandeng_images if os.path.exists(img)]
    
    print(f"\n📸 Found {len(existing_bandeng_images)} bandeng images")
    
    if len(existing_bandeng_images) == 0:
        print("❌ No bandeng images found")
        return
    
    # Use subset for training
    training_images = existing_bandeng_images[:4]  # Use first 4 for training
    test_images = existing_bandeng_images[4:] if len(existing_bandeng_images) > 4 else existing_bandeng_images[:2]  # Use remaining for testing
    
    # Add bandeng with extensive augmentation
    print(f"\n🔄 Adding Ikan Bandeng with extensive augmentation...")
    print(f"📚 Training images: {len(training_images)}")
    
    add_results = system.add_species_with_extensive_augmentation(
        training_images,
        "Ikan Bandeng"
    )
    
    if add_results['success']:
        print(f"✅ Successfully added Ikan Bandeng!")
        print(f"   Total embeddings: {add_results['total_embeddings_added']}")
        
        # Test recognition
        print(f"\n🧪 Testing recognition...")
        print(f"🧩 Test images: {len(test_images)}")
        
        test_results = system.test_recognition_accuracy(test_images, "Ikan Bandeng")
        
        print(f"\n📊 Final Results:")
        print(f"   Recognition Accuracy: {test_results['accuracy']:.1f}%")
        print(f"   Total Training Embeddings: {add_results['total_embeddings_added']}")
        
        if test_results['accuracy'] >= 50:
            print("🎉 SUCCESS! Extensive preprocessing significantly improved recognition!")
        elif test_results['accuracy'] >= 25:
            print("✅ GOOD! Noticeable improvement achieved!")
        else:
            print("⚠️ Still needs more augmentation or different approach")
            
        # Show individual predictions
        print(f"\n🔍 Individual Test Results:")
        for pred in test_results['predictions']:
            status = "✅" if pred['is_correct'] else "❌"
            print(f"   {status} {pred['image']}: {pred['predicted_species']} (conf: {pred['confidence']:.3f})")
    else:
        print("❌ Failed to add species")

if __name__ == "__main__":
    main()