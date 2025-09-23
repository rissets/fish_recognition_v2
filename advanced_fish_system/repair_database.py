#!/usr/bin/env python3
"""
Database repair script to fix the keys mapping issue
for Mujair embeddings.
"""

import torch
import json
import os
from datetime import datetime

def repair_database():
    """Repair the database keys mapping"""
    print("🔧 REPAIRING DATABASE KEYS MAPPING")
    print("=" * 50)
    
    db_path = "../models/classification/database.pt"
    backup_path = f"../models/classification/backup/database_repair_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    
    # Load database
    database = torch.load(db_path, map_location='cpu')
    embeddings, internal_ids, image_ids, annotation_ids, drawn_fish_ids, keys = database
    
    print(f"📊 Current state:")
    print(f"   Total embeddings: {len(embeddings)}")
    print(f"   Keys in dictionary: {len(keys)}")
    
    # Find Mujair key
    mujair_key = None
    for k, v in keys.items():
        if isinstance(v, dict) and 'label' in v and 'Mujair' in v['label']:
            mujair_key = k
            break
        elif isinstance(v, str) and 'Mujair' in v:
            mujair_key = k
            break
    
    if mujair_key is None:
        print("❌ No Mujair key found in database")
        return False
    
    print(f"✅ Found Mujair at key: {mujair_key}")
    
    # Create backup
    os.makedirs(os.path.dirname(backup_path), exist_ok=True)
    torch.save(database, backup_path)
    print(f"💾 Backup created: {backup_path}")
    
    # Analyze internal_ids and identify which ones should map to Mujair
    # Based on our analysis, all internal_ids from around 69995+ should be Mujair
    original_embeddings_count = len([k for k in keys.keys() if isinstance(k, str)])
    print(f"📈 Original species count: {original_embeddings_count}")
    
    # Count how many embeddings should belong to Mujair
    # We added ~65 Mujair embeddings, so let's find them
    mujair_internal_ids = []
    
    # Strategy: Find all internal_ids that don't have corresponding keys
    for i, internal_id in enumerate(internal_ids):
        if internal_id not in keys and str(internal_id) not in keys:
            mujair_internal_ids.append((i, internal_id))
    
    print(f"🔍 Found {len(mujair_internal_ids)} embeddings without keys mapping")
    print(f"   Range: {mujair_internal_ids[0][1] if mujair_internal_ids else 'N/A'} to {mujair_internal_ids[-1][1] if mujair_internal_ids else 'N/A'}")
    
    if not mujair_internal_ids:
        print("✅ No repair needed - all embeddings have keys")
        return True
    
    # Create a new keys structure that properly maps internal_ids to species
    # Convert the dictionary-based keys to a list-based structure
    print(f"🔄 Converting keys structure...")
    
    # Create a list where index = internal_id and value = species_name
    max_internal_id = max(internal_ids)
    new_keys = [''] * (max_internal_id + 1)
    
    # Fill in original species mappings
    for key_id, species_info in keys.items():
        if isinstance(species_info, dict) and 'label' in species_info:
            species_name = species_info['label']
        elif isinstance(species_info, str):
            species_name = species_info
        else:
            continue
            
        # Map all embeddings that belong to this species
        for i, internal_id in enumerate(internal_ids):
            if str(internal_id) == str(key_id) or internal_id == key_id:
                new_keys[internal_id] = species_name
    
    # Map all orphaned embeddings to Mujair
    mujair_species_name = "Ikan Mujair"
    mapped_count = 0
    for i, internal_id in mujair_internal_ids:
        new_keys[internal_id] = mujair_species_name
        mapped_count += 1
    
    print(f"✅ Mapped {mapped_count} embeddings to {mujair_species_name}")
    
    # Verify the mapping
    mujair_count = sum(1 for key in new_keys if key == mujair_species_name)
    print(f"📊 Total Mujair embeddings after repair: {mujair_count}")
    
    # Save repaired database
    repaired_database = [embeddings, internal_ids, image_ids, annotation_ids, drawn_fish_ids, new_keys]
    torch.save(repaired_database, db_path)
    print(f"💾 Repaired database saved")
    
    print(f"\n✅ DATABASE REPAIR COMPLETE")
    print(f"   Mapped {mapped_count} orphaned embeddings to Mujair")
    print(f"   Total Mujair embeddings: {mujair_count}")
    return True

if __name__ == "__main__":
    repair_database()