#!/usr/bin/env python3
"""
🖼️ Test Full Image Processing (No Cropping)
"""

import os
import sys
import time
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_full_image_processing():
    """Test preprocessing with full image (no cropping)"""
    
    print("="*60)
    print("🖼️ TESTING FULL IMAGE PROCESSING (NO CROPPING)")
    print("="*60)
    
    try:
        from advanced_preprocessing import AdvancedFishPreprocessor
        
        # Initialize with full image mode
        preprocessor_full = AdvancedFishPreprocessor(use_full_image=True)
        preprocessor_crop = AdvancedFishPreprocessor(use_full_image=False)
        
        logger.info("✅ Initialized preprocessors (full image & cropped)")
    except ImportError as e:
        logger.error(f"❌ Failed to import: {e}")
        return False
    
    # Find test image
    test_image_paths = [
        "../images/bandeng.jpg",
        "../images/mujair1.jpg"
    ]
    
    test_image = None
    for path in test_image_paths:
        if os.path.exists(path):
            test_image = path
            break
    
    if not test_image:
        logger.error("❌ No test image found")
        return False
    
    logger.info(f"🖼️ Testing with: {test_image}")
    
    # Test 1: Full image processing
    print("\n🖼️ TEST 1: Full Image Processing (No Cropping)")
    print("-" * 50)
    
    try:
        start_time = time.time()
        results_full = preprocessor_full.process_single_image(
            test_image,
            output_dir="output/full_image_test"
        )
        processing_time = time.time() - start_time
        
        if results_full:
            logger.info(f"✅ Full image processing completed in {processing_time:.2f}s")
            logger.info(f"📊 Generated {len(results_full)} full image versions")
        else:
            logger.error("❌ Full image processing failed")
            
    except Exception as e:
        logger.error(f"❌ Full image processing error: {e}")
    
    # Test 2: Cropped processing (for comparison)
    print("\n✂️ TEST 2: Cropped Processing (For Comparison)")
    print("-" * 50)
    
    try:
        start_time = time.time()
        results_crop = preprocessor_crop.process_single_image(
            test_image,
            output_dir="output/cropped_test"
        )
        processing_time = time.time() - start_time
        
        if results_crop:
            logger.info(f"✅ Cropped processing completed in {processing_time:.2f}s")
            logger.info(f"📊 Generated {len(results_crop)} cropped versions")
        else:
            logger.error("❌ Cropped processing failed")
            
    except Exception as e:
        logger.error(f"❌ Cropped processing error: {e}")
    
    # Compare file sizes
    print("\n📏 TEST 3: Size Comparison")
    print("-" * 50)
    
    full_dir = "output/full_image_test"
    crop_dir = "output/cropped_test"
    
    if os.path.exists(full_dir) and os.path.exists(crop_dir):
        full_files = [f for f in os.listdir(full_dir) if f.endswith('.jpg')]
        crop_files = [f for f in os.listdir(crop_dir) if f.endswith('.jpg')]
        
        if full_files and crop_files:
            # Check first file from each
            import cv2
            
            full_img = cv2.imread(os.path.join(full_dir, full_files[0]))
            crop_img = cv2.imread(os.path.join(crop_dir, crop_files[0]))
            
            if full_img is not None and crop_img is not None:
                full_h, full_w = full_img.shape[:2]
                crop_h, crop_w = crop_img.shape[:2]
                
                logger.info(f"📐 Full image size: {full_w}x{full_h}")
                logger.info(f"📐 Cropped size: {crop_w}x{crop_h}")
                logger.info(f"📊 Size ratio: {(crop_w*crop_h)/(full_w*full_h):.2%}")
    
    # Summary
    print("\n" + "="*60)
    print("🎯 COMPARISON SUMMARY")
    print("="*60)
    print("🖼️ Full Image: Preserves complete fish without cropping")
    print("✂️ Cropped: Focuses on detected fish region")
    print("💡 Recommendation: Use full image to avoid cropping issues")
    print("="*60)
    
    return True

def main():
    """Main function"""
    try:
        # Change to script directory
        script_dir = Path(__file__).parent
        os.chdir(script_dir)
        
        # Run test
        success = test_full_image_processing()
        
        if success:
            print("\n🎉 Full image test completed!")
            print("\n📁 Check output folders:")
            print("   • output/full_image_test/ (full images)")
            print("   • output/cropped_test/ (cropped images)")
            print("\n💡 To use full image mode in other scripts:")
            print("   preprocessor = AdvancedFishPreprocessor(use_full_image=True)")
        else:
            print("\n❌ Test failed")
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()