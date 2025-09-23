#!/usr/bin/env python3
"""
Deep debug script to analyze Unknown classification issue
by directly inspecting classification results and database.
"""

import sys
import os
sys.path.append('..')
from advanced_fish_recognition import AdvancedFishSystem
import cv2
import torch
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_database():
    """Analyze current database state"""
    print("🔍 ANALYZING DATABASE STATE")
    print("=" * 50)
    
    db_path = "../models/classification/database.pt"
    labels_path = "../models/classification/labels.json"
    
    if not os.path.exists(db_path):
        print("❌ Database file not found")
        return
    
    # Load database
    database = torch.load(db_path, map_location='cpu')
    if len(database) != 6:
        print(f"❌ Invalid database structure: {len(database)} components")
        return
    
    embeddings, internal_ids, image_ids, annotation_ids, drawn_fish_ids, keys = database
    
    print(f"📊 Database Statistics:")
    print(f"   Total embeddings: {len(embeddings)}")
    print(f"   Embedding dimension: {embeddings.shape[1] if len(embeddings) > 0 else 'N/A'}")
    print(f"   Total keys: {len(keys)}")
    print(f"   Total internal IDs: {len(internal_ids)}")
    
    # Load labels
    if os.path.exists(labels_path):
        with open(labels_path, 'r', encoding='utf-8') as f:
            labels = json.load(f)
        print(f"   Total labels: {len(labels)}")
        
        # Check for Mujair
        mujair_count = 0
        mujair_key = "Ikan Mujair"
        for key in keys:
            if key == mujair_key:
                mujair_count += 1
        
        print(f"   Mujair embeddings: {mujair_count}")
        print(f"   Mujair percentage: {(mujair_count/len(keys)*100):.3f}%")
        
        # Show last 10 entries
        print(f"\n📋 Last 10 database entries:")
        try:
            for i in range(max(0, len(keys)-10), len(keys)):
                if i < len(keys):
                    print(f"      {i}: {keys[i]}")
                else:
                    print(f"      {i}: INDEX OUT OF RANGE")
        except Exception as e:
            print(f"      Error showing entries: {e}")
            print(f"      Keys type: {type(keys)}")
            print(f"      Keys length: {len(keys) if hasattr(keys, '__len__') else 'No length'}")
            if hasattr(keys, '__getitem__'):
                try:
                    print(f"      Last key: {keys[-1] if len(keys) > 0 else 'No keys'}")
                except:
                    print(f"      Cannot access last key")
    
    print()

def debug_classification_direct():
    """Debug classification by calling classifier directly"""
    print("🔍 DIRECT CLASSIFICATION DEBUG")
    print("=" * 50)
    
    # Initialize system
    system = AdvancedFishSystem()
    
    test_images = [
        "../images/mujair1.jpg",
        "../images/mujair2.jpg", 
        "../images/mujair3.jpg"
    ]
    
    for i, image_path in enumerate(test_images, 1):
        print(f"\n📸 Testing image {i}: {os.path.basename(image_path)}")
        print("-" * 40)
        
        try:
            # Load and detect fish
            img = cv2.imread(image_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Detection
            detection_results = system.detector.predict(img_rgb)
            if not detection_results or len(detection_results) == 0 or len(detection_results[0]) == 0:
                print("   ❌ No fish detected")
                continue
            
            best_detection = detection_results[0][0]
            detection_conf = float(best_detection.score)
            print(f"   ✅ Fish detected (conf: {detection_conf:.3f})")
            
            # Get cropped fish
            cropped_bgr = best_detection.get_mask_BGR()
            print(f"   📏 Cropped image shape: {cropped_bgr.shape}")
            
            # Direct classification call
            print(f"   🧠 Calling batch_inference...")
            classification_results = system.classifier.batch_inference([cropped_bgr])
            
            if classification_results and len(classification_results) > 0:
                result = classification_results[0]
                print(f"   📊 Classification result type: {type(result)}")
                print(f"   📊 Classification result length: {len(result) if result else 0}")
                
                if result and len(result) > 0:
                    top_result = result[0]
                    print(f"   🎯 Top result: {top_result}")
                    
                    # Show top 5 results
                    print(f"   📈 Top 5 predictions:")
                    for j, pred in enumerate(result[:5], 1):
                        print(f"      {j}. {pred['name']} (acc: {pred['accuracy']:.3f})")
                        
                    # Check if top result is Unknown_XXXXX
                    if top_result['name'].startswith('Unknown_'):
                        internal_id = top_result['name'].split('_')[1]
                        print(f"   ⚠️ UNKNOWN DETECTION - Internal ID: {internal_id}")
                        
                        # Check database for this ID
                        db_path = "../models/classification/database.pt"
                        database = torch.load(db_path, map_location='cpu')
                        keys = database[5]
                        
                        try:
                            actual_species = keys[int(internal_id)]
                            print(f"   🔍 Actual species for ID {internal_id}: '{actual_species}'")
                        except:
                            print(f"   ❌ Cannot find species for ID {internal_id}")
                else:
                    print("   ❌ No classification results")
            else:
                print("   ❌ Classification failed")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Main debug function"""
    print("🚀 COMPREHENSIVE UNKNOWN CLASSIFICATION DEBUG")
    print("=" * 60)
    
    # First analyze database
    analyze_database()
    
    # Then test classification directly
    debug_classification_direct()
    
    print("\n" + "=" * 60)
    print("🏁 DEBUG COMPLETE")

if __name__ == "__main__":
    main()