#!/usr/bin/env python3
"""
Clean restart: Restore original database and properly add Mujair embeddings
"""

import torch
import json
import os
from datetime import datetime

def clean_restart_database():
    """Clean restart with original database and proper Mujair addition"""
    print("🔄 CLEAN DATABASE RESTART")
    print("=" * 50)
    
    db_path = "../models/classification/database.pt"
    
    # Restore from clean backup (before our modifications)
    clean_backup = "../models/classification/backup/database_backup_20250923_235436.pt"
    
    if not os.path.exists(clean_backup):
        print(f"❌ Clean backup not found: {clean_backup}")
        return False
    
    print(f"📥 Restoring clean database from: {clean_backup}")
    
    # Load clean backup
    clean_data = torch.load(clean_backup, map_location='cpu')
    
    # Save current as backup
    current_backup = f"../models/classification/backup/database_before_clean_restart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    if os.path.exists(db_path):
        current_data = torch.load(db_path, map_location='cpu')
        torch.save(current_data, current_backup)
        print(f"💾 Current database backed up to: {current_backup}")
    
    # Restore clean database
    torch.save(clean_data, db_path)
    
    # Analyze clean database
    embeddings, internal_ids, image_ids, annotation_ids, drawn_fish_ids, keys = clean_data
    
    print(f"📊 Clean database state:")
    print(f"   Total embeddings: {len(embeddings)}")
    print(f"   Keys count: {len(keys)}")
    print(f"   Keys type: {type(keys)}")
    
    # Check if Mujair already exists
    mujair_found = False
    mujair_id = None
    
    for key, value in keys.items():
        if isinstance(value, dict) and 'label' in value:
            if 'Mujair' in value['label']:
                mujair_found = True
                mujair_id = key
                break
        elif isinstance(value, str) and 'Mujair' in value:
            mujair_found = True
            mujair_id = key
            break
    
    if mujair_found:
        print(f"✅ Mujair already exists in clean database with ID: {mujair_id}")
        
        # Count existing Mujair embeddings
        mujair_count = sum(1 for internal_id in internal_ids if internal_id == mujair_id)
        print(f"📊 Existing Mujair embeddings: {mujair_count}")
    else:
        print("ℹ️ Mujair not found in clean database")
        
        # Add Mujair species to keys
        numeric_keys = [k for k in keys.keys() if isinstance(k, (int, float))]
        string_keys = [k for k in keys.keys() if isinstance(k, str) and k.isdigit()]
        
        all_numeric_keys = numeric_keys + [int(k) for k in string_keys]
        max_id = max(all_numeric_keys) if all_numeric_keys else 427
        
        mujair_id = max_id + 1
        keys[mujair_id] = {
            'label': 'Ikan Mujair',
            'added_date': datetime.now().isoformat(),
            'source': 'manual'
        }
        print(f"✅ Added Mujair species with ID: {mujair_id}")
        
        # Save updated database
        updated_data = [embeddings, internal_ids, image_ids, annotation_ids, drawn_fish_ids, keys]
        torch.save(updated_data, db_path)
    
    print(f"\n✅ CLEAN RESTART COMPLETE")
    print(f"   Database restored to clean state")
    print(f"   Mujair species ID: {mujair_id}")
    print(f"   Ready for fresh Mujair embedding addition")
    
    return True

if __name__ == "__main__":
    clean_restart_database()