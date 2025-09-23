#!/usr/bin/env python3
"""
Debug script to understand why classification returns Unknown_XXXXX
instead of species names after aggressive boosting.
"""

import sys
import os
sys.path.append('..')
from advanced_fish_recognition import AdvancedFishSystem
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_unknown_classification():
    """Debug the Unknown classification issue"""
    print("🔍 DEBUGGING UNKNOWN CLASSIFICATION ISSUE")
    print("=" * 60)
    
    # Initialize system
    print("\n1️⃣ Initializing system...")
    system = AdvancedFishSystem()
    
    # Test paths
    test_images = [
        "../images/mujair1.jpg",
        "../images/mujair2.jpg", 
        "../images/mujair3.jpg"
    ]
    
    print(f"\n2️⃣ Testing {len(test_images)} images...")
    
    for i, image_path in enumerate(test_images, 1):
        print(f"\n📸 Testing image {i}: {os.path.basename(image_path)}")
        print("-" * 40)
        
        try:
            # Get detailed prediction
            result = system.predict_fish_complete(image_path)
            
            if result and 'classification' in result:
                classification = result['classification']
                
                print(f"   🎯 Final Result: {classification['predicted_species']} (conf: {classification['confidence']:.3f})")
                
                # Check if it's an Unknown prediction
                if classification['predicted_species'].startswith('Unknown_'):
                    print(f"   ⚠️ UNKNOWN DETECTION: {classification['predicted_species']}")
                    
                    # Try to extract the internal ID
                    try:
                        internal_id = classification['predicted_species'].split('_')[1]
                        print(f"   📍 Internal ID: {internal_id}")
                        
                        # Load database to check what species this ID corresponds to
                        import torch
                        db_path = "../models/classification/database.pt"
                        if os.path.exists(db_path):
                            database = torch.load(db_path, map_location='cpu')
                            if len(database) >= 6:
                                embeddings, internal_ids, image_ids, annotation_ids, drawn_fish_ids, keys = database
                                
                                # Find this internal ID in the database
                                internal_id_int = int(internal_id)
                                if internal_id_int < len(keys):
                                    actual_species = keys[internal_id_int]
                                    print(f"   🔍 Database lookup: ID {internal_id} = '{actual_species}'")
                                else:
                                    print(f"   ❌ Internal ID {internal_id} not found in database")
                            else:
                                print(f"   ❌ Invalid database structure")
                        else:
                            print(f"   ❌ Database file not found")
                            
                    except Exception as e:
                        print(f"   ❌ Error parsing Unknown ID: {e}")
                
                # Show top predictions if available
                if 'top_predictions' in classification:
                    print(f"   📊 Top 3 predictions:")
                    for j, pred in enumerate(classification['top_predictions'][:3], 1):
                        print(f"      {j}. {pred['species']} (conf: {pred['confidence']:.3f})")
                        
            else:
                print(f"   ❌ No classification result")
                
        except Exception as e:
            print(f"   ❌ Error processing {image_path}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("🔍 DEBUG ANALYSIS COMPLETE")

if __name__ == "__main__":
    debug_unknown_classification()