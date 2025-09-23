#!/usr/bin/env python3
"""
Proper database repair that maintains classifier compatibility
"""

import torch
import json
import os
from datetime import datetime

def repair_database_properly():
    """Properly repair database while maintaining classifier compatibility"""
    print("🔧 PROPER DATABASE REPAIR")
    print("=" * 50)
    
    # First, restore from backup if the current database is broken
    db_path = "../models/classification/database.pt"
    
    # Find the most recent backup before our repair
    backup_dir = "../models/classification/backup"
    backup_files = [f for f in os.listdir(backup_dir) if f.startswith("database_backup_") and f.endswith(".pt")]
    backup_files.sort(reverse=True)
    
    # Use the backup from before aggressive boosting
    restore_backup = None
    for backup in backup_files:
        if "database_backup_20250924_000250" in backup:  # From aggressive boosting
            restore_backup = os.path.join(backup_dir, backup)
            break
    
    if not restore_backup and backup_files:
        restore_backup = os.path.join(backup_dir, backup_files[0])
    
    if restore_backup:
        print(f"📥 Restoring from backup: {restore_backup}")
        backup_data = torch.load(restore_backup, map_location='cpu')
        torch.save(backup_data, db_path)
    
    # Load the database
    database = torch.load(db_path, map_location='cpu')
    embeddings, internal_ids, image_ids, annotation_ids, drawn_fish_ids, keys = database
    
    print(f"📊 Current state after restore:")
    print(f"   Total embeddings: {len(embeddings)}")
    print(f"   Keys type: {type(keys)}")
    print(f"   Keys count: {len(keys)}")
    
    # Ensure keys is a dictionary
    if not isinstance(keys, dict):
        print("❌ Keys is not a dictionary - cannot repair")
        return False
    
    # Find Mujair species ID
    mujair_species_id = None
    for species_id, species_info in keys.items():
        if isinstance(species_info, dict) and 'label' in species_info:
            if 'Mujair' in species_info['label']:
                mujair_species_id = species_id
                break
        elif isinstance(species_info, str) and 'Mujair' in species_info:
            mujair_species_id = species_id
            break
    
    if mujair_species_id is None:
        print("❌ Mujair species not found in keys")
        return False
    
    print(f"✅ Found Mujair species ID: {mujair_species_id}")
    
    # Create backup before repair
    backup_path = f"../models/classification/backup/database_proper_repair_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    torch.save(database, backup_path)
    print(f"💾 Backup created: {backup_path}")
    
    # Strategy: Fix internal_ids to map all Mujair embeddings to the correct species_id
    # Identify which embeddings should be Mujair (the recent additions)
    
    # Count original embeddings vs current embeddings
    original_species_count = len([k for k in keys.keys() if isinstance(k, (int, str))])
    total_embeddings = len(embeddings)
    
    print(f"📈 Analysis:")
    print(f"   Original species: {original_species_count}")
    print(f"   Total embeddings: {total_embeddings}")
    
    # Find embeddings that don't have proper species mapping
    # These should be the Mujair embeddings we added
    
    # Check internal_ids distribution
    id_counts = {}
    for internal_id in internal_ids:
        id_counts[internal_id] = id_counts.get(internal_id, 0) + 1
    
    # Find internal_ids that don't exist in keys
    orphaned_indices = []
    for i, internal_id in enumerate(internal_ids):
        if internal_id not in keys:
            orphaned_indices.append(i)
    
    print(f"🔍 Found {len(orphaned_indices)} orphaned embeddings")
    
    if len(orphaned_indices) > 0:
        print(f"   Range: indices {orphaned_indices[0]} to {orphaned_indices[-1]}")
        print(f"   Internal IDs: {internal_ids[orphaned_indices[0]]} to {internal_ids[orphaned_indices[-1]]}")
        
        # Map all orphaned embeddings to Mujair species ID
        for idx in orphaned_indices:
            internal_ids[idx] = mujair_species_id
        
        print(f"✅ Mapped {len(orphaned_indices)} embeddings to Mujair (species ID: {mujair_species_id})")
    
    # Verify the repair
    mujair_count = sum(1 for internal_id in internal_ids if internal_id == mujair_species_id)
    print(f"📊 Total Mujair embeddings after repair: {mujair_count}")
    
    # Save repaired database
    repaired_database = [embeddings, internal_ids, image_ids, annotation_ids, drawn_fish_ids, keys]
    torch.save(repaired_database, db_path)
    print(f"💾 Repaired database saved")
    
    print(f"\n✅ PROPER DATABASE REPAIR COMPLETE")
    print(f"   Mapped {len(orphaned_indices)} orphaned embeddings to Mujair")
    print(f"   Total Mujair embeddings: {mujair_count}")
    return True

if __name__ == "__main__":
    repair_database_properly()