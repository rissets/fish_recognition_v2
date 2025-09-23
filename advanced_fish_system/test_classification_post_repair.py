#!/usr/bin/env python3
"""
Test classification after database repair
"""

import sys
import os
sys.path.append('..')
from advanced_fish_recognition import AdvancedFishSystem
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_classification_post_repair():
    """Test classification after database repair"""
    print("🧪 TESTING CLASSIFICATION AFTER DATABASE REPAIR")
    print("=" * 60)
    
    # Initialize system (this will reload the repaired database)
    print("\n1️⃣ Initializing system with repaired database...")
    system = AdvancedFishSystem()
    
    test_images = [
        "../images/mujair1.jpg",
        "../images/mujair2.jpg", 
        "../images/mujair3.jpg"
    ]
    
    print(f"\n2️⃣ Testing {len(test_images)} images...")
    
    success_count = 0
    total_count = len(test_images)
    
    for i, image_path in enumerate(test_images, 1):
        print(f"\n📸 Testing image {i}: {os.path.basename(image_path)}")
        print("-" * 40)
        
        try:
            result = system.predict_fish_complete(image_path)
            
            if result:
                species = result['species']
                confidence = result['confidence']
                
                print(f"   🎯 Result: {species} (conf: {confidence:.3f})")
                
                if 'Mujair' in species:
                    print(f"   ✅ SUCCESS! Correctly identified as Mujair")
                    success_count += 1
                elif species.startswith('Unknown_'):
                    print(f"   ⚠️ Still Unknown prediction")
                else:
                    print(f"   ❌ Wrong species prediction")
            else:
                print(f"   ❌ No prediction result")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Calculate success rate
    success_rate = (success_count / total_count) * 100
    
    print(f"\n" + "=" * 60)
    print(f"📊 FINAL RESULTS:")
    print(f"   Successful predictions: {success_count}/{total_count}")
    print(f"   Success rate: {success_rate:.1f}%")
    
    if success_rate >= 66.7:  # At least 2/3 correct
        print(f"   🎉 EXCELLENT! Classification is working!")
    elif success_rate >= 33.3:  # At least 1/3 correct
        print(f"   ✅ GOOD! Significant improvement achieved!")
    else:
        print(f"   ⚠️ Still needs improvement")
    
    return success_rate

if __name__ == "__main__":
    test_classification_post_repair()