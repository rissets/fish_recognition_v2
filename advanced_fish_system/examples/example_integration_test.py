#!/usr/bin/env python3
"""
Example: Complete Integration Test
Test lengkap semua fitur advanced fish system
"""

import sys
import os
from pathlib import Path
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from advanced_fish_recognition import AdvancedFishSystem
from database_utilities import DatabaseUtilities
from batch_processor import BatchProcessor

def test_system_initialization():
    """Test system initialization"""
    print("🔧 Testing System Initialization...")
    
    system = AdvancedFishSystem()
    
    if system.initialized:
        print("✅ System initialized successfully!")
        
        # Test individual components
        if hasattr(system, 'detector') and system.detector:
            print("  ✅ Detection model loaded")
        else:
            print("  ❌ Detection model failed to load")
            
        if hasattr(system, 'classifier') and system.classifier:
            print("  ✅ Classification model loaded")
        else:
            print("  ❌ Classification model failed to load")
            
        if hasattr(system, 'segmentor') and system.segmentor:
            print("  ✅ Segmentation model loaded")
        else:
            print("  ❌ Segmentation model failed to load")
            
        return system
    else:
        print("❌ System initialization failed!")
        return None

def test_database_utilities():
    """Test database utilities"""
    print("\n💾 Testing Database Utilities...")
    
    database_path = "../models/classification/database.pt"
    labels_path = "../models/classification/labels.json"
    
    utils = DatabaseUtilities(database_path, labels_path)
    
    # Test stats
    try:
        stats = utils.get_database_stats()
        print(f"  ✅ Database stats retrieved: {stats['total_embeddings']} embeddings")
    except Exception as e:
        print(f"  ❌ Failed to get database stats: {e}")
        return False
    
    # Test integrity
    try:
        integrity = utils.validate_database_integrity()
        if integrity['valid']:
            print("  ✅ Database integrity is valid")
        else:
            print(f"  ⚠️ Database has {len(integrity['issues'])} integrity issues")
    except Exception as e:
        print(f"  ❌ Failed to validate integrity: {e}")
        return False
    
    # Test backup
    try:
        backup_path = utils.create_backup("integration_test")
        if backup_path:
            print(f"  ✅ Backup created successfully")
        else:
            print("  ❌ Failed to create backup")
    except Exception as e:
        print(f"  ❌ Backup creation failed: {e}")
        return False
    
    return True

def test_fish_addition(system):
    """Test adding fish to database"""
    print("\n🐟 Testing Fish Addition...")
    
    # Test image path
    test_image = Path("../../images/mujair1.jpg")
    
    if not test_image.exists():
        print(f"  ❌ Test image not found: {test_image}")
        print("  Please ensure test image exists for complete testing")
        return False
    
    # Get initial stats
    initial_stats = system.get_database_stats()
    initial_count = initial_stats['total_embeddings']
    
    try:
        # Add fish to database
        success = system.add_fish_to_database_permanent(
            image_path=str(test_image),
            species_name="Test Mujair"
        )
        
        if success:
            # Verify addition
            final_stats = system.get_database_stats()
            final_count = final_stats['total_embeddings']
            
            if final_count > initial_count:
                print(f"  ✅ Fish added successfully! ({final_count - initial_count} new embeddings)")
                return True
            else:
                print("  ❌ Fish addition reported success but no new embeddings found")
                return False
        else:
            print("  ❌ Fish addition failed")
            return False
            
    except Exception as e:
        print(f"  ❌ Fish addition error: {e}")
        return False

def test_prediction(system):
    """Test fish prediction"""
    print("\n🔍 Testing Fish Prediction...")
    
    test_image = Path("../../images/mujair1.jpg")
    
    if not test_image.exists():
        print(f"  ❌ Test image not found: {test_image}")
        return False
    
    try:
        result = system.predict_fish_complete(str(test_image))
        
        if result:
            print(f"  ✅ Prediction successful!")
            print(f"    - Species: {result['species']}")
            print(f"    - Confidence: {result['confidence']:.3f}")
            print(f"    - Detection confidence: {result['detection_confidence']:.3f}")
            print(f"    - Has mask: {result['has_mask']}")
            return True
        else:
            print("  ❌ Prediction failed")
            return False
            
    except Exception as e:
        print(f"  ❌ Prediction error: {e}")
        return False

def test_batch_processing():
    """Test batch processing"""
    print("\n📦 Testing Batch Processing...")
    
    try:
        processor = BatchProcessor()
        
        if not processor.system.initialized:
            print("  ❌ Batch processor initialization failed")
            return False
        
        # Test with images folder
        images_folder = Path("../../images")
        
        if images_folder.exists():
            result = processor.process_species_folder(
                species_folder=images_folder,
                species_name="Test Batch Species",
                max_images_per_species=3  # Limit for testing
            )
            
            print(f"  ✅ Batch processing completed!")
            print(f"    - Total images: {result.total_images}")
            print(f"    - Successful: {result.successful}")
            print(f"    - Success rate: {result.success_rate:.1f}%")
            return True
        else:
            print(f"  ⚠️ Images folder not found: {images_folder}")
            print("  Skipping batch processing test")
            return True
            
    except Exception as e:
        print(f"  ❌ Batch processing error: {e}")
        return False

def test_error_handling(system):
    """Test error handling"""
    print("\n⚠️ Testing Error Handling...")
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Invalid image path
    try:
        result = system.add_fish_to_database_permanent(
            image_path="nonexistent_image.jpg",
            species_name="Test Species"
        )
        if not result:  # Should fail gracefully
            print("  ✅ Invalid image path handled correctly")
            tests_passed += 1
        else:
            print("  ❌ Invalid image path should have failed")
    except Exception as e:
        print(f"  ❌ Invalid image path caused exception: {e}")
    
    # Test 2: Empty species name
    try:
        test_image = Path("../../images/mujair1.jpg")
        if test_image.exists():
            result = system.add_fish_to_database_permanent(
                image_path=str(test_image),
                species_name=""
            )
            if not result:  # Should fail gracefully
                print("  ✅ Empty species name handled correctly")
                tests_passed += 1
            else:
                print("  ❌ Empty species name should have failed")
        else:
            print("  ⚠️ Skipping empty species name test - no test image")
            tests_passed += 1  # Skip this test
    except Exception as e:
        print(f"  ❌ Empty species name caused exception: {e}")
    
    # Test 3: Invalid prediction path
    try:
        result = system.predict_fish_complete("nonexistent_prediction_image.jpg")
        if not result:  # Should fail gracefully
            print("  ✅ Invalid prediction path handled correctly")
            tests_passed += 1
        else:
            print("  ❌ Invalid prediction path should have failed")
    except Exception as e:
        print(f"  ❌ Invalid prediction path caused exception: {e}")
    
    return tests_passed == total_tests

def run_integration_test():
    """Run complete integration test"""
    print("=" * 60)
    print("🧪 ADVANCED FISH SYSTEM INTEGRATION TEST")
    print("=" * 60)
    
    test_results = []
    start_time = time.time()
    
    # Test 1: System Initialization
    system = test_system_initialization()
    test_results.append(("System Initialization", system is not None))
    
    if system is None:
        print("\n❌ Cannot continue tests - system initialization failed")
        return
    
    # Test 2: Database Utilities
    db_utils_ok = test_database_utilities()
    test_results.append(("Database Utilities", db_utils_ok))
    
    # Test 3: Fish Addition
    fish_addition_ok = test_fish_addition(system)
    test_results.append(("Fish Addition", fish_addition_ok))
    
    # Test 4: Prediction
    prediction_ok = test_prediction(system)
    test_results.append(("Fish Prediction", prediction_ok))
    
    # Test 5: Batch Processing
    batch_ok = test_batch_processing()
    test_results.append(("Batch Processing", batch_ok))
    
    # Test 6: Error Handling
    error_handling_ok = test_error_handling(system)
    test_results.append(("Error Handling", error_handling_ok))
    
    # Summary
    total_time = time.time() - start_time
    passed_tests = sum(1 for _, result in test_results if result)
    total_tests = len(test_results)
    
    print("\n" + "=" * 60)
    print("📊 INTEGRATION TEST SUMMARY")
    print("=" * 60)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<25} : {status}")
    
    print(f"\nResults: {passed_tests}/{total_tests} tests passed")
    print(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")
    print(f"Total time: {total_time:.2f} seconds")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED! System is working correctly.")
    elif passed_tests >= total_tests * 0.8:
        print("\n⚠️ Most tests passed. System is mostly functional.")
    else:
        print("\n❌ Multiple test failures. System needs attention.")
    
    # Final database stats
    print("\n📊 Final Database Statistics:")
    final_stats = system.get_database_stats()
    print(f"  - Total embeddings: {final_stats['total_embeddings']}")
    print(f"  - Total species: {final_stats['total_labels']}")
    
    print("\n✅ Integration test completed!")

if __name__ == "__main__":
    run_integration_test()