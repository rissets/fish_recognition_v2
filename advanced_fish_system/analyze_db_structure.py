#!/usr/bin/env python3
"""
Database structure analysis to understand the keys mapping issue
"""

import torch
import json

def analyze_database_structure():
    """Deep analysis of database structure"""
    print("🔍 DEEP DATABASE STRUCTURE ANALYSIS")
    print("=" * 60)
    
    db_path = "../models/classification/database.pt"
    database = torch.load(db_path, map_location='cpu')
    
    print(f"📊 Database components: {len(database)}")
    for i, component in enumerate(database):
        print(f"   {i}: {type(component)} - Length: {len(component) if hasattr(component, '__len__') else 'N/A'}")
    
    embeddings, internal_ids, image_ids, annotation_ids, drawn_fish_ids, keys = database
    
    print(f"\n📈 Component Details:")
    print(f"   Embeddings shape: {embeddings.shape}")
    print(f"   Internal IDs type: {type(internal_ids)}, length: {len(internal_ids)}")
    print(f"   Image IDs type: {type(image_ids)}, length: {len(image_ids)}")
    print(f"   Annotation IDs type: {type(annotation_ids)}, length: {len(annotation_ids)}")
    print(f"   Drawn Fish IDs type: {type(drawn_fish_ids)}, length: {len(drawn_fish_ids)}")
    print(f"   Keys type: {type(keys)}, length: {len(keys)}")
    
    # Analyze internal_ids
    print(f"\n🔍 Internal IDs Analysis:")
    if hasattr(internal_ids, '__getitem__'):
        print(f"   First 5 internal IDs: {internal_ids[:5] if len(internal_ids) >= 5 else internal_ids}")
        print(f"   Last 5 internal IDs: {internal_ids[-5:] if len(internal_ids) >= 5 else internal_ids}")
        print(f"   Internal IDs range: {min(internal_ids)} to {max(internal_ids)}")
    
    # Analyze keys
    print(f"\n🗝️ Keys Analysis:")
    if isinstance(keys, dict):
        print(f"   Keys is a dictionary with {len(keys)} entries")
        try:
            numeric_keys = [k for k in keys.keys() if isinstance(k, (int, float))]
            string_keys = [k for k in keys.keys() if isinstance(k, str)]
            print(f"   Numeric keys: {len(numeric_keys)}")
            print(f"   String keys: {len(string_keys)}")
            
            if numeric_keys:
                print(f"   Numeric key range: {min(numeric_keys)} to {max(numeric_keys)}")
            if string_keys:
                print(f"   String keys sample: {string_keys[:5]}")
                
        except Exception as e:
            print(f"   Error analyzing key types: {e}")
        
        print(f"   Sample key-value pairs:")
        for i, (k, v) in enumerate(list(keys.items())[:10]):
            print(f"      {k} ({type(k).__name__}): {v}")
        
        # Check for Mujair in values
        mujair_entries = [(k, v) for k, v in keys.items() if 'Mujair' in str(v)]
        print(f"   Mujair entries: {mujair_entries}")
        
    elif isinstance(keys, list):
        print(f"   Keys is a list with {len(keys)} entries")
        print(f"   First 5 keys: {keys[:5] if len(keys) >= 5 else keys}")
        print(f"   Last 5 keys: {keys[-5:] if len(keys) >= 5 else keys}")
        
        # Check for Mujair in list
        mujair_indices = [i for i, key in enumerate(keys) if 'Mujair' in str(key)]
        print(f"   Mujair indices: {mujair_indices}")
    
    # Check the relationship between internal_ids and keys
    print(f"\n🔗 ID-Key Relationship Analysis:")
    if len(internal_ids) > 0:
        print(f"   Looking for internal ID {internal_ids[-1]} in keys...")
        if isinstance(keys, dict):
            if internal_ids[-1] in keys:
                print(f"   Found: {keys[internal_ids[-1]]}")
            else:
                print(f"   NOT FOUND in dictionary keys")
        elif isinstance(keys, list):
            if internal_ids[-1] < len(keys):
                print(f"   Found at index {internal_ids[-1]}: {keys[internal_ids[-1]]}")
            else:
                print(f"   INDEX OUT OF RANGE (max index: {len(keys)-1})")
    
    # Check recent additions
    print(f"\n🆕 Recent Additions Analysis:")
    print(f"   Recent internal IDs (last 10): {internal_ids[-10:] if len(internal_ids) >= 10 else internal_ids}")
    
    # Try to find what species IDs 70048-70061 correspond to
    target_ids = [70048, 70049, 70050, 70061]
    for target_id in target_ids:
        print(f"   Checking ID {target_id}:")
        if target_id in internal_ids:
            idx = list(internal_ids).index(target_id)
            print(f"     Found at embedding index {idx}")
            if isinstance(keys, dict) and target_id in keys:
                print(f"     Species: {keys[target_id]}")
            elif isinstance(keys, list) and target_id < len(keys):
                print(f"     Species: {keys[target_id]}")
            else:
                print(f"     Species: NOT FOUND in keys")
        else:
            print(f"     NOT FOUND in internal_ids")

if __name__ == "__main__":
    analyze_database_structure()