#!/usr/bin/env python3
"""
Quick Start Script
Script cepat untuk testing Advanced Fish System
"""

import sys
import os
from pathlib import Path
import argparse

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from advanced_fish_recognition import AdvancedFishSystem
from database_utilities import DatabaseUtilities

def quick_test():
    """Quick test of the system"""
    print("🚀 Quick Start - Advanced Fish System")
    print("=" * 50)
    
    # Initialize system
    print("🔧 Initializing system...")
    system = AdvancedFishSystem()
    
    if not system.initialized:
        print("❌ System initialization failed!")
        print("Please check:")
        print("  1. Model files exist in ../models/ folders")
        print("  2. Database file exists at ../models/classification/database.pt")
        print("  3. Labels file exists at ../models/classification/labels.json")
        return False
    
    print("✅ System initialized successfully!")
    
    # Show database stats
    print("\n📊 Database Statistics:")
    stats = system.get_database_stats()
    print(f"  - Total embeddings: {stats['total_embeddings']}")
    print(f"  - Total species: {stats['total_labels']}")
    print(f"  - Embedding dimension: {stats['embedding_dimension']}")
    
    # Test with available images
    test_images = [
        Path("../../images/mujair1.jpg"),
        Path("../../images/mujair2.jpg"),
        Path("../../images/bandeng.jpg"),
        Path("../image_testing/mujair4.jpg")
    ]
    
    found_image = None
    for img_path in test_images:
        if img_path.exists():
            found_image = img_path
            break
    
    if found_image:
        print(f"\n🔍 Testing with image: {found_image}")
        
        # Test prediction
        result = system.predict_fish_complete(str(found_image))
        
        if result:
            print("✅ Prediction successful!")
            print(f"  - Species: {result['species']}")
            print(f"  - Confidence: {result['confidence']:.3f}")
            print(f"  - Detection confidence: {result['detection_confidence']:.3f}")
            print(f"  - Has mask: {result['has_mask']}")
        else:
            print("❌ Prediction failed")
    else:
        print("\n⚠️ No test images found")
        print("Available test images should be at:")
        for img_path in test_images:
            print(f"  - {img_path}")
    
    print("\n🎉 Quick test completed!")
    print("\nNext steps:")
    print("  1. Run full integration test: python examples/example_integration_test.py")
    print("  2. Add fish to database: python examples/example_basic_addition.py")
    print("  3. Batch process images: python examples/example_batch_processing.py")
    print("  4. Database management: python examples/example_database_management.py")
    
    return True

def add_fish_interactive():
    """Interactive fish addition"""
    print("🐟 Interactive Fish Addition")
    print("=" * 30)
    
    # Initialize system
    system = AdvancedFishSystem()
    
    if not system.initialized:
        print("❌ System initialization failed!")
        return False
    
    # Get image path from user
    while True:
        image_path = input("\n📷 Enter fish image path (or 'quit' to exit): ").strip()
        
        if image_path.lower() == 'quit':
            break
            
        if not Path(image_path).exists():
            print(f"❌ Image not found: {image_path}")
            continue
        
        # Get species name
        species_name = input("🏷️ Enter species name: ").strip()
        
        if not species_name:
            print("❌ Species name cannot be empty")
            continue
        
        # Add to database
        print("🔄 Adding fish to database...")
        success = system.add_fish_to_database_permanent(image_path, species_name)
        
        if success:
            print("✅ Fish added successfully!")
            
            # Test prediction
            result = system.predict_fish_complete(image_path)
            if result:
                print(f"  - Prediction: {result['species']} (confidence: {result['confidence']:.3f})")
        else:
            print("❌ Failed to add fish")
        
        # Ask to continue
        continue_choice = input("\n➕ Add another fish? (y/n): ").strip().lower()
        if continue_choice != 'y':
            break
    
    print("👋 Interactive session ended")
    return True

def database_info():
    """Show database information"""
    print("📊 Database Information")
    print("=" * 25)
    
    database_path = "../models/classification/database.pt"
    labels_path = "../models/classification/labels.json"
    
    utils = DatabaseUtilities(database_path, labels_path)
    
    if not utils.database_path.exists():
        print(f"❌ Database not found: {database_path}")
        return False
    
    # Stats
    stats = utils.get_database_stats()
    print(f"Database file: {utils.database_path}")
    print(f"Size: {stats['database_size_mb']:.2f} MB")
    print(f"Last modified: {stats['last_modified']}")
    print(f"Total embeddings: {stats['total_embeddings']}")
    print(f"Embedding dimension: {stats['embedding_dimension']}")
    print(f"Total species: {stats['total_labels']}")
    
    # Integrity check
    print("\n🔍 Integrity Check:")
    integrity = utils.validate_database_integrity()
    
    if integrity['valid']:
        print("✅ Database integrity is VALID")
    else:
        print("⚠️ Database integrity issues:")
        for issue in integrity['issues']:
            print(f"  - {issue}")
    
    # Backups
    backups = utils.list_backups()
    print(f"\n💾 Available backups: {len(backups)}")
    if backups:
        for backup in backups[:5]:
            print(f"  - {backup['filename']} ({backup['size_mb']:.2f} MB)")
    
    return True

def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(description="Advanced Fish System Quick Start")
    parser.add_argument('--mode', choices=['test', 'add', 'info'], default='test',
                       help='Mode: test (quick test), add (interactive addition), info (database info)')
    
    args = parser.parse_args()
    
    print(f"🐟 Advanced Fish Recognition System")
    print(f"Mode: {args.mode}")
    print()
    
    if args.mode == 'test':
        success = quick_test()
    elif args.mode == 'add':
        success = add_fish_interactive()
    elif args.mode == 'info':
        success = database_info()
    else:
        print("❌ Invalid mode")
        success = False
    
    if success:
        print("\n✅ Operation completed successfully!")
    else:
        print("\n❌ Operation failed!")
    
    return success

if __name__ == "__main__":
    main()