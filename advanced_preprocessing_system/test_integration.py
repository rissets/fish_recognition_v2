#!/usr/bin/env python3
"""
🧪 Integration Test Script
Tests the complete system with existing models
"""

import os
import sys
import logging
from pathlib import Path

# Setup logging  
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_integration():
    """Test integration with existing models"""
    
    print("🔬 TESTING INTEGRATION WITH EXISTING MODELS")
    print("="*60)
    
    # Test model paths
    models_base = os.path.join(os.path.dirname(__file__), '..', 'models')
    
    model_paths = {
        "detection": os.path.join(models_base, 'detection', 'model.ts'),
        "segmentation": os.path.join(models_base, 'segmentation', 'model.ts'),
        "classification": os.path.join(models_base, 'classification', 'model.ts'),
        "database": os.path.join(models_base, 'classification', 'database.pt'),
        "labels": os.path.join(models_base, 'classification', 'labels.json')
    }
    
    # Check model availability
    for model_name, path in model_paths.items():
        if os.path.exists(path):
            logger.info(f"✅ {model_name} model found: {path}")
        else:
            logger.warning(f"⚠️  {model_name} model not found: {path}")
    
    # Test imports
    try:
        from advanced_preprocessing import AdvancedFishPreprocessor
        preprocessor = AdvancedFishPreprocessor()
        
        logger.info(f"🎯 Detection model: {'✅ Loaded' if preprocessor.detector else '❌ Failed'}")
        logger.info(f"🎯 Segmentation model: {'✅ Loaded' if preprocessor.segmentator else '❌ Failed'}")
        logger.info(f"🎯 Classification model: {'✅ Loaded' if preprocessor.classifier else '❌ Failed'}")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize preprocessor: {e}")
        return False
    
    # Test with sample image if available
    test_image = None
    for test_path in ["../images/bandeng.jpg", "../images/mujair1.jpg"]:
        if os.path.exists(test_path):
            test_image = test_path
            break
    
    if test_image:
        logger.info(f"🖼️  Testing with image: {test_image}")
        
        try:
            results = preprocessor.process_single_image(
                test_image,
                output_dir="output/integration_test"
            )
            
            if results:
                logger.info(f"✅ Integration test successful: {len(results)} images generated")
                if isinstance(results, dict):
                    for strategy in results.keys():
                        logger.info(f"   • {strategy}")
                else:
                    logger.info(f"   • Generated {len(results)} processed images")
            else:
                logger.error("❌ Integration test failed")
                
        except Exception as e:
            logger.error(f"❌ Integration test error: {e}")
    else:
        logger.warning("⚠️  No test images found for integration test")
    
    print("\n🏁 Integration test completed!")
    return True

if __name__ == "__main__":
    test_model_integration()