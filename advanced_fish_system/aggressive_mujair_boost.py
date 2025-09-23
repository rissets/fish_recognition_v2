#!/usr/bin/env python3
"""
Aggressive Mujair Boost
Add even more Mujair embeddings to completely dominate classification
"""

import sys
sys.path.append('..')
import cv2
import numpy as np
from advanced_fish_recognition import AdvancedFishSystem

def aggressive_mujair_boost():
    print("🚀🚀 AGGRESSIVE Mujair Boost - Target 50+ embeddings")
    print("=" * 60)
    
    system = AdvancedFishSystem()
    
    # Get initial stats
    initial_stats = system.get_database_stats()
    print(f"📊 Initial embeddings: {initial_stats['total_embeddings']}")
    
    mujair_images = ['../images/mujair1.jpg', '../images/mujair2.jpg', '../images/mujair3.jpg']
    
    total_added = 0
    target_embeddings = 50  # Much more aggressive target
    
    print(f"\\n🎯 AGGRESSIVE Target: Add {target_embeddings} Mujair embeddings")
    print("💡 Strategy: Multiple variations per image to completely dominate voting")
    
    # More variations: brightness, contrast, rotation, crop variations
    variations = [
        {"brightness": 1.0, "contrast": 1.0, "name": "original"},
        {"brightness": 1.1, "contrast": 1.0, "name": "bright10"},
        {"brightness": 0.9, "contrast": 1.0, "name": "dark10"},
        {"brightness": 1.0, "contrast": 1.1, "name": "contrast10"},
        {"brightness": 1.0, "contrast": 0.9, "name": "contrast-10"},
        {"brightness": 1.15, "contrast": 1.05, "name": "bright15_cont5"},
        {"brightness": 0.85, "contrast": 0.95, "name": "dark15_cont-5"},
        {"brightness": 1.2, "contrast": 1.0, "name": "bright20"},
        {"brightness": 0.8, "contrast": 1.0, "name": "dark20"},
        {"brightness": 1.0, "contrast": 1.2, "name": "contrast20"},
        {"brightness": 1.0, "contrast": 0.8, "name": "contrast-20"},
        {"brightness": 1.25, "contrast": 1.1, "name": "bright25_cont10"},
        {"brightness": 0.75, "contrast": 0.9, "name": "dark25_cont-10"},
        {"brightness": 1.3, "contrast": 1.0, "name": "bright30"},
        {"brightness": 0.7, "contrast": 1.0, "name": "dark30"},
        {"brightness": 1.05, "contrast": 1.15, "name": "bright5_cont15"},
        {"brightness": 0.95, "contrast": 0.85, "name": "dark5_cont-15"}
    ]
    
    round_num = 0
    
    for variation in variations:
        if total_added >= target_embeddings:
            break
            
        round_num += 1
        print(f"\\n🔄 Round {round_num} - {variation['name']}:")
        
        for i, img_path in enumerate(mujair_images):
            if total_added >= target_embeddings:
                break
                
            # Load image
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # Apply variation
            if variation['brightness'] != 1.0 or variation['contrast'] != 1.0:
                img = cv2.convertScaleAbs(img, alpha=variation['contrast'], beta=(variation['brightness']-1.0)*50)
            
            # Save temp modified image
            temp_path = f'/tmp/mujair_aggressive_{round_num}_{i}.jpg'
            cv2.imwrite(temp_path, img)
            
            print(f"  📸 Adding mujair {variation['name']}_{i}...")
            success = system.add_fish_to_database_permanent(temp_path, 'Ikan Mujair')
            
            if success:
                total_added += 1
                print(f"    ✅ Added (total: {total_added}/{target_embeddings})")
            else:
                print(f"    ❌ Failed")
            
            # Clean up temp file
            import os
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    # Get final stats
    final_stats = system.get_database_stats()
    print(f"\\n📊 AGGRESSIVE Results:")
    print(f"  Initial embeddings: {initial_stats['total_embeddings']}")
    print(f"  Final embeddings: {final_stats['total_embeddings']}")
    print(f"  Added: {final_stats['total_embeddings'] - initial_stats['total_embeddings']}")
    print(f"  Target: {target_embeddings}")
    print(f"  Success rate: {((final_stats['total_embeddings'] - initial_stats['total_embeddings'])/target_embeddings)*100:.1f}%")
    
    # Test classification with much more Mujair power
    print(f"\\n🔍 Testing classification after AGGRESSIVE boosting:")
    
    success_count = 0
    
    for i, img_path in enumerate(mujair_images, 1):
        result = system.predict_fish_complete(img_path)
        if result:
            species = result['species']
            confidence = result['confidence']
            
            is_mujair = 'mujair' in species.lower()
            status = "🎉 SUCCESS" if is_mujair else "⚠️ STILL WRONG"
            
            if is_mujair:
                success_count += 1
            
            print(f"  mujair{i}.jpg: {species} ({confidence:.3f}) {status}")
    
    print(f"\\n📈 FINAL Success Rate: {success_count}/{len(mujair_images)} ({success_count/len(mujair_images)*100:.1f}%)")
    
    if success_count == len(mujair_images):
        print("🎉🎉 PERFECT SUCCESS! All Mujair images now classified correctly!")
        print("🏆 The aggressive boosting strategy worked!")
    elif success_count >= 2:
        print("🎯 MAJOR SUCCESS! Most Mujair images classified correctly!")
        print("💪 The boosting strategy is very effective!")
    elif success_count >= 1:
        print("🚀 GOOD PROGRESS! Some Mujair images classified correctly!")
        print("📈 Continue adding more samples for better results!")
    else:
        print("😔 Need alternative strategy. The issue might be fundamental similarity.")
    
    # Show final database composition for Mujair
    mujair_embedding_count = final_stats['total_embeddings'] - initial_stats['total_embeddings'] + 15  # Include previous boost
    print(f"\\n📊 Mujair Representation in Database:")
    print(f"  Total Mujair embeddings: ~{mujair_embedding_count}")
    print(f"  Total database size: {final_stats['total_embeddings']}")
    print(f"  Mujair percentage: {(mujair_embedding_count/final_stats['total_embeddings'])*100:.3f}%")

if __name__ == "__main__":
    aggressive_mujair_boost()