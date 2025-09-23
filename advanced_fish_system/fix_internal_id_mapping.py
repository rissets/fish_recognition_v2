#!/usr/bin/env python3
"""
Fix Internal ID Mapping Issue
The problem is internal_ids don't match with labels.json mapping
"""

import sys
sys.path.append('..')
import torch
import json
from advanced_fish_recognition import AdvancedFishSystem

def fix_internal_id_mapping():
    print("🔧 Fix Internal ID Mapping Issue")
    print("=" * 45)
    
    # Load database
    db_data = torch.load('../models/classification/database.pt')
    embeddings = db_data[0]
    internal_ids = db_data[1]
    image_ids = db_data[2]
    annotation_ids = db_data[3]
    drawn_fish_ids = db_data[4]
    keys = db_data[5]
    
    # Load labels
    with open('../models/classification/labels.json', 'r', encoding='utf-8') as f:
        labels = json.load(f)
    
    print(f"📊 Database info:")
    print(f"  Total embeddings: {len(embeddings)}")
    print(f"  Total internal_ids: {len(internal_ids)}")
    print(f"  Total labels: {len(labels)}")
    
    # Check last 10 internal_ids
    print(f"\n🔍 Last 10 internal_ids:")
    for i in range(max(0, len(internal_ids)-10), len(internal_ids)):
        internal_id = internal_ids[i]
        print(f"  Index {i}: internal_id = {internal_id}")
    
    # Check what species ID 427 should be (Ikan Mujair)
    print(f"\n🐟 Mujair info:")
    print(f"  Labels['427'] = {labels.get('427', 'NOT FOUND')}")
    
    # Check keys
    print(f"\n🗝️ Keys info:")
    print(f"  Keys length: {len(keys)}")
    if 427 in keys:
        print(f"  Keys[427] = {keys[427]}")
    
    # The problem: Last 3 internal_ids should be 427 for Mujair
    print(f"\n⚠️ Problem Analysis:")
    print(f"  Last 3 internal_ids: {internal_ids[-3:]}")
    print(f"  These should all be 427 for 'Ikan Mujair'")
    
    # Fix: Update internal_ids for the last 3 embeddings
    print(f"\n🔧 Fixing internal_ids...")
    
    # Create backup
    backup_path = '../models/classification/database_backup_before_fix.pt'
    torch.save(db_data, backup_path)
    print(f"✅ Backup created: {backup_path}")
    
    # Fix internal_ids - set last 3 to 427 (Mujair species ID)
    new_internal_ids = internal_ids.copy()
    for i in range(3):
        idx = len(new_internal_ids) - 3 + i
        new_internal_ids[idx] = 427
        print(f"  Fixed index {idx}: {internal_ids[idx]} → 427")
    
    # Create new database with fixed internal_ids
    new_db_data = [
        embeddings,
        new_internal_ids,
        image_ids,
        annotation_ids,
        drawn_fish_ids,
        keys
    ]
    
    # Save fixed database
    torch.save(new_db_data, '../models/classification/database.pt')
    print(f"✅ Database saved with fixed internal_ids")
    
    # Test the fix
    print(f"\n🧪 Testing the fix...")
    
    system = AdvancedFishSystem()
    if system.initialized:
        print("✅ System reloaded")
        
        # Test prediction
        result = system.predict_fish_complete('../images/mujair1.jpg')
        if result:
            print(f"\n🔍 Prediction result:")
            print(f"  Species: {result['species']}")
            print(f"  Confidence: {result['confidence']:.3f}")
            
            if 'mujair' in result['species'].lower():
                print("🎉 SUCCESS: Now correctly predicting Mujair!")
            else:
                print("⚠️ Still not Mujair, but let's check all mujair images...")
                
                # Test all mujair images
                for i, img_path in enumerate(['../images/mujair1.jpg', '../images/mujair2.jpg', '../images/mujair3.jpg'], 1):
                    result = system.predict_fish_complete(img_path)
                    if result:
                        print(f"  mujair{i}.jpg: {result['species']} ({result['confidence']:.3f})")
    
    print(f"\n✅ Fix completed!")

if __name__ == "__main__":
    fix_internal_id_mapping()