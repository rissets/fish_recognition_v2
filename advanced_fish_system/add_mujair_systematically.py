#!/usr/bin/env python3
"""
Add Mujair images to clean database systematically
"""

import sys
import os
sys.path.append('..')
from advanced_fish_recognition import AdvancedFishSystem
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def add_mujair_systematically():
    """Add Mujair images to database systematically"""
    print("🐟 SYSTEMATIC MUJAIR ADDITION")
    print("=" * 50)
    
    # Initialize system
    system = AdvancedFishSystem()
    
    # Test images
    mujair_images = [
        "../images/mujair1.jpg",
        "../images/mujair2.jpg", 
        "../images/mujair3.jpg"
    ]
    
    success_count = 0
    
    for i, image_path in enumerate(mujair_images, 1):
        print(f"\n📸 Adding Mujair image {i}: {os.path.basename(image_path)}")
        print("-" * 30)
        
        try:
            result = system.add_fish_to_database_permanent(image_path, "Ikan Mujair")
            
            if result:
                print(f"   ✅ Successfully added!")
                success_count += 1
            else:
                print(f"   ❌ Failed to add")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n📊 Addition Results:")
    print(f"   Successfully added: {success_count}/{len(mujair_images)}")
    
    # Test classification after addition
    if success_count > 0:
        print(f"\n🧪 Testing classification after addition...")
        
        for i, image_path in enumerate(mujair_images, 1):
            try:
                result = system.predict_fish_complete(image_path)
                if result:
                    species = result['species']
                    confidence = result['confidence']
                    
                    print(f"   {i}. {os.path.basename(image_path)}: {species} (conf: {confidence:.3f})")
                    
                    if 'Mujair' in species:
                        print(f"      ✅ SUCCESS!")
                    else:
                        print(f"      ❌ Still wrong")
                        
            except Exception as e:
                print(f"   {i}. Error testing {os.path.basename(image_path)}: {e}")

if __name__ == "__main__":
    add_mujair_systematically()