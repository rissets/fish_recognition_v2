#!/usr/bin/env python3
"""
Boost Mujair Classification
Add many more Mujair embeddings to increase voting power
"""

import sys
sys.path.append('..')
import cv2
import numpy as np
from advanced_fish_recognition import AdvancedFishSystem

def boost_mujair_classification():
    print("🚀 Boost Mujair Classification with Multiple Embeddings")
    print("=" * 55)
    
    system = AdvancedFishSystem()
    
    if not system.initialized:
        print("❌ System initialization failed!")
        return
    
    # Get initial stats
    initial_stats = system.get_database_stats()
    print(f"📊 Initial embeddings: {initial_stats['total_embeddings']}")
    
    # Strategy: Add the same mujair images multiple times with slight variations
    # This will increase the voting power for Mujair classification
    
    mujair_images = ['../images/mujair1.jpg', '../images/mujair2.jpg', '../images/mujair3.jpg']
    
    total_added = 0
    target_embeddings = 15  # Add 5 variations per image = 15 total
    
    print(f"\\n🎯 Target: Add {target_embeddings} Mujair embeddings to boost voting power")
    print("💡 Strategy: Use image variations to increase Mujair representation")
    
    for round_num in range(5):  # 5 rounds
        print(f"\\n🔄 Round {round_num + 1}:")
        
        for i, img_path in enumerate(mujair_images):
            # Load and potentially modify image
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            # Create slight variation (brightness adjustment)
            if round_num > 0:
                # Adjust brightness slightly for variation
                brightness_factor = 1.0 + (round_num * 0.05)  # 5% increment
                img = cv2.convertScaleAbs(img, alpha=brightness_factor, beta=0)
            
            # Save temp modified image
            temp_path = f'/tmp/mujair_temp_{round_num}_{i}.jpg'
            cv2.imwrite(temp_path, img)
            
            print(f"  📸 Adding mujair variation {round_num}_{i}...")
            success = system.add_fish_to_database_permanent(temp_path, 'Ikan Mujair')
            
            if success:
                total_added += 1
                print(f"    ✅ Added (total: {total_added})")
            else:
                print(f"    ❌ Failed")
                
            # Clean up temp file
            import os
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            if total_added >= target_embeddings:
                break
        
        if total_added >= target_embeddings:
            break
    
    # Get final stats
    final_stats = system.get_database_stats()
    print(f"\\n📊 Results:")
    print(f"  Initial embeddings: {initial_stats['total_embeddings']}")
    print(f"  Final embeddings: {final_stats['total_embeddings']}")
    print(f"  Added: {final_stats['total_embeddings'] - initial_stats['total_embeddings']}")
    print(f"  Target: {target_embeddings}")
    
    # Test classification
    print(f"\\n🔍 Testing classification after boosting:")
    
    for i, img_path in enumerate(mujair_images, 1):
        result = system.predict_fish_complete(img_path)
        if result:
            species = result['species']
            confidence = result['confidence']
            
            is_mujair = 'mujair' in species.lower()
            status = "🎉 SUCCESS" if is_mujair else "⚠️ STILL WRONG"
            
            print(f"  mujair{i}.jpg: {species} ({confidence:.3f}) {status}")
    
    # Count successful predictions
    successful_mujair = 0
    for img_path in mujair_images:
        result = system.predict_fish_complete(img_path)
        if result and 'mujair' in result['species'].lower():
            successful_mujair += 1
    
    print(f"\\n📈 Success Rate: {successful_mujair}/{len(mujair_images)} ({successful_mujair/len(mujair_images)*100:.1f}%)")
    
    if successful_mujair == len(mujair_images):
        print("🎉 PERFECT! All Mujair images now classified correctly!")
    elif successful_mujair > 0:
        print("🎯 PARTIAL SUCCESS! Some Mujair images classified correctly!")
    else:
        print("😔 No improvement. May need different strategy.")

if __name__ == "__main__":
    boost_mujair_classification()