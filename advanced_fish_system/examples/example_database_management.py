#!/usr/bin/env python3
"""
Example: Database Management
Contoh penggunaan database utilities untuk maintenance
"""

import sys
import os
from pathlib import Path
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from database_utilities import DatabaseUtilities
from advanced_fish_recognition import AdvancedFishSystem

def main():
    """Example: Database management and utilities"""
    
    print("=== Database Management Example ===")
    print()
    
    # Initialize database utilities
    print("🔧 Initializing Database Utilities...")
    database_path = "../models/classification/database.pt"
    labels_path = "../models/classification/labels.json"
    
    utils = DatabaseUtilities(database_path, labels_path)
    
    if not utils.database_path.exists():
        print(f"❌ Database not found: {database_path}")
        print("Please make sure the database path is correct")
        return
    
    print("✅ Database utilities initialized successfully!")
    print()
    
    # 1. Database Statistics
    print("📊 Database Statistics:")
    print("-" * 30)
    stats = utils.get_database_stats()
    print(f"  - Total embeddings: {stats['total_embeddings']}")
    print(f"  - Embedding dimension: {stats['embedding_dimension']}")
    print(f"  - Total species: {stats['total_labels']}")
    print(f"  - Database file size: {stats['database_size_mb']:.2f} MB")
    print(f"  - Last modified: {stats['last_modified']}")
    print()
    
    # 2. Integrity Validation
    print("🔍 Database Integrity Validation:")
    print("-" * 35)
    integrity = utils.validate_database_integrity()
    
    if integrity['valid']:
        print("✅ Database integrity is VALID")
        print("  - All components have consistent lengths")
        print("  - No duplicates detected")
        print("  - Labels file is consistent with database")
    else:
        print("⚠️ Database integrity issues detected:")
        for issue in integrity['issues']:
            print(f"  - ❌ {issue}")
        
        print("\nRecommendations:")
        for rec in integrity['recommendations']:
            print(f"  - 💡 {rec}")
        
        print("\n🔧 Attempting automatic repair...")
        repair_success = utils.repair_database()
        
        if repair_success:
            print("✅ Database repair completed successfully!")
            
            # Re-validate after repair
            integrity_after = utils.validate_database_integrity()
            if integrity_after['valid']:
                print("✅ Database integrity is now VALID after repair")
            else:
                print("⚠️ Some issues may still exist after repair")
        else:
            print("❌ Database repair failed")
    print()
    
    # 3. Backup Management
    print("💾 Backup Management:")
    print("-" * 20)
    
    # Create a backup
    print("Creating backup...")
    backup_path = utils.create_backup("example_backup")
    if backup_path:
        print(f"✅ Backup created: {backup_path}")
    else:
        print("❌ Failed to create backup")
    
    # List all backups
    print("\nAvailable backups:")
    backups = utils.list_backups()
    if backups:
        for i, backup in enumerate(backups[:5]):  # Show first 5
            print(f"  {i+1}. {backup['filename']} ({backup['size_mb']:.2f} MB)")
        if len(backups) > 5:
            print(f"  ... and {len(backups) - 5} more backups")
    else:
        print("  No backups found")
    
    # Clean old backups (keep 10 most recent)
    print(f"\nCleaning old backups (keeping 10 most recent)...")
    deleted_count = utils.clean_old_backups(keep_count=10)
    print(f"✅ Deleted {deleted_count} old backup(s)")
    print()
    
    # 4. Advanced Analysis
    print("🔬 Advanced Database Analysis:")
    print("-" * 30)
    
    # Initialize system for advanced analysis
    system = AdvancedFishSystem()
    if system.initialized:
        # Get species distribution
        stats = system.get_database_stats()
        
        print(f"Available species (showing first 10):")
        species_list = stats.get('available_species', [])
        for i, species in enumerate(species_list[:10]):
            print(f"  {i+1}. {species}")
        if len(species_list) > 10:
            print(f"  ... and {len(species_list) - 10} more species")
        print()
        
        # Embedding statistics
        if hasattr(system.classifier, 'database') and system.classifier.database:
            embeddings = system.classifier.database[0]  # First element is embeddings
            if len(embeddings) > 0:
                import torch
                
                # Calculate embedding statistics
                norms = torch.norm(embeddings, dim=1)
                mean_norm = torch.mean(norms).item()
                std_norm = torch.std(norms).item()
                min_norm = torch.min(norms).item()
                max_norm = torch.max(norms).item()
                
                print("📈 Embedding Statistics:")
                print(f"  - Mean norm: {mean_norm:.3f}")
                print(f"  - Std norm: {std_norm:.3f}")
                print(f"  - Min norm: {min_norm:.3f}")
                print(f"  - Max norm: {max_norm:.3f}")
                print()
    
    # 5. Database Optimization Tips
    print("💡 Database Optimization Tips:")
    print("-" * 30)
    print("✅ Regular maintenance checklist:")
    print("  1. Run integrity validation weekly")
    print("  2. Create backups before major operations")
    print("  3. Clean old backups monthly")
    print("  4. Monitor embedding statistics for anomalies")
    print("  5. Keep labels.json in sync with database")
    print()
    
    print("🔧 Performance optimization:")
    print("  1. Remove duplicate embeddings if any")
    print("  2. Ensure consistent data types")
    print("  3. Validate embedding dimensions")
    print("  4. Check for corrupted entries")
    print()
    
    # 6. Recovery Example
    print("🚨 Emergency Recovery Example:")
    print("-" * 30)
    print("If database becomes corrupted:")
    print("  1. Stop all operations immediately")
    print("  2. Run: integrity = utils.validate_database_integrity()")
    print("  3. If issues found, run: utils.repair_database()")
    print("  4. If repair fails, restore from latest backup:")
    print("     utils.restore_from_backup(backup_path)")
    print("  5. Re-validate integrity after recovery")
    print()
    
    print("✅ Database management example completed!")

if __name__ == "__main__":
    main()