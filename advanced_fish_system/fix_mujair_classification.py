#!/usr/bin/env python3
"""
Fix Mujair Classification
Add multiple mujair samples to improve classification accuracy
"""

import sys
sys.path.append('..')
from advanced_fish_recognition import AdvancedFishSystem

def main():
    print("🔧 Fix Mujair Classification")
    print("=" * 40)
    
    # Initialize system
    system = AdvancedFishSystem()
    
    if not system.initialized:
        print("❌ System initialization failed!")
        return
    
    print("✅ System initialized")
    
    # Get initial stats
    print("\n📊 Initial stats:")
    initial_stats = system.get_database_stats()
    print(f"Total embeddings: {initial_stats['total_embeddings']}")
    
    # Test current prediction
    print("\n🔍 Current prediction for mujair1.jpg:")
    result = system.predict_fish_complete('../images/mujair1.jpg')
    if result:
        print(f"Species: {result['species']}")
        print(f"Confidence: {result['confidence']:.3f}")
    
    # Add more mujair samples
    print("\n🐟 Adding more mujair samples...")
    
    mujair_images = ['../images/mujair2.jpg', '../images/mujair3.jpg']
    success_count = 0
    
    for i, img_path in enumerate(mujair_images, 1):
        print(f"\n📸 Adding mujair image {i}: {img_path}")
        success = system.add_fish_to_database_permanent(img_path, 'Ikan Mujair')
        
        if success:
            print(f"✅ Successfully added")
            success_count += 1
        else:
            print(f"❌ Failed to add")
    
    print(f"\n📊 Added {success_count}/{len(mujair_images)} mujair samples")
    
    # Get updated stats
    final_stats = system.get_database_stats()
    print(f"Total embeddings: {final_stats['total_embeddings']}")
    print(f"New embeddings added: {final_stats['total_embeddings'] - initial_stats['total_embeddings']}")
    
    # Test prediction again
    print("\n🔍 Testing prediction after adding more samples:")
    result = system.predict_fish_complete('../images/mujair1.jpg')
    if result:
        print(f"Species: {result['species']}")
        print(f"Confidence: {result['confidence']:.3f}")
        
        if 'mujair' in result['species'].lower():
            print("🎉 SUCCESS: Now correctly predicting Mujair!")
        else:
            print("⚠️ Still not predicting Mujair correctly")
            print("💡 Suggestion: May need more samples or the images might be too similar to Nila")
    
    # Test all mujair images
    print("\n🧪 Testing all mujair images:")
    for i, img_path in enumerate(['../images/mujair1.jpg', '../images/mujair2.jpg', '../images/mujair3.jpg'], 1):
        print(f"\n📷 Testing {img_path}:")
        result = system.predict_fish_complete(img_path)
        if result:
            print(f"  Species: {result['species']}")
            print(f"  Confidence: {result['confidence']:.3f}")

if __name__ == "__main__":
    main()