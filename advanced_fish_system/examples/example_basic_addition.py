#!/usr/bin/env python3
"""
Example: Basic Fish Addition
Contoh dasar menambah ikan ke database secara permanen
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from advanced_fish_recognition import AdvancedFishSystem

def main():
    """Example: Add single fish to database permanently"""
    
    print("=== Basic Fish Addition Example ===")
    print()
    
    # Initialize advanced fish recognition system
    print("🔧 Initializing Advanced Fish System...")
    system = AdvancedFishSystem()
    
    if not system.initialized:
        print("❌ Failed to initialize system")
        return
    
    print("✅ System initialized successfully!")
    print()
    
    # Get initial database stats
    print("📊 Initial Database Statistics:")
    stats = system.get_database_stats()
    print(f"  - Total embeddings: {stats['total_embeddings']}")
    print(f"  - Total species: {stats['total_labels']}")
    print(f"  - Embedding dimension: {stats['embedding_dimension']}")
    print()
    
    # Test image path - adjust to your test image
    test_image = Path("../images/mujair1.jpg")
    
    if not test_image.exists():
        print(f"❌ Test image not found: {test_image}")
        print("Please adjust the test_image path to point to your fish image")
        return
    
    print(f"🐟 Adding fish from: {test_image}")
    print("Species name: Ikan Mujair")
    print()
    
    # Add fish to database permanently
    print("🔄 Processing fish addition...")
    success = system.add_fish_to_database_permanent(
        image_path=str(test_image),
        species_name="Ikan Mujair"
    )
    
    if success:
        print("✅ Fish added successfully to database!")
        print()
        
        # Get updated stats
        print("📊 Updated Database Statistics:")
        updated_stats = system.get_database_stats()
        print(f"  - Total embeddings: {updated_stats['total_embeddings']}")
        print(f"  - Total species: {updated_stats['total_labels']}")
        print(f"  - New embeddings added: {updated_stats['total_embeddings'] - stats['total_embeddings']}")
        print()
        
        # Test prediction on the same image
        print("🔍 Testing prediction on the added fish...")
        prediction_result = system.predict_fish_complete(str(test_image))
        
        if prediction_result:
            print("✅ Prediction successful!")
            print(f"  - Detected species: {prediction_result['species']}")
            print(f"  - Confidence: {prediction_result['confidence']:.3f}")
            print(f"  - Detection confidence: {prediction_result['detection_confidence']:.3f}")
            print(f"  - Has mask: {prediction_result['has_mask']}")
        else:
            print("❌ Prediction failed")
    
    else:
        print("❌ Failed to add fish to database")
        print("Check the logs for error details")

if __name__ == "__main__":
    main()