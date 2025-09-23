#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Database Utilities untuk Advanced Fish Recognition System
Tools untuk mengelola, backup, dan maintain database
"""

import os
import torch
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseUtilities:
    """Utility class untuk operasi database lanjutan"""
    
    def __init__(self, db_path: str, labels_path: str):
        self.db_path = Path(db_path)
        self.labels_path = Path(labels_path)
        self.backup_dir = self.db_path.parent / "backup"
        self.backup_dir.mkdir(exist_ok=True)
    
    def validate_database_integrity(self) -> Dict[str, Any]:
        """Validasi integritas database secara menyeluruh"""
        logger.info("🔍 Validating database integrity...")
        
        try:
            if not self.db_path.exists():
                return {
                    'valid': False,
                    'error': 'Database file not found',
                    'issues': [],
                    'recommendations': ['Create new database']
                }
            
            # Load database
            db_data = torch.load(self.db_path)
            
            issues = []
            recommendations = []
            
            # Check structure
            if not isinstance(db_data, (list, tuple)) or len(db_data) != 6:
                issues.append(f"Invalid structure: expected 6 elements, got {len(db_data) if hasattr(db_data, '__len__') else 'unknown'}")
                recommendations.append("Recreate database with correct structure")
                return {
                    'valid': False,
                    'error': 'Invalid database structure',
                    'issues': issues,
                    'recommendations': recommendations
                }
            
            embeddings, internal_ids, image_ids, annotation_ids, drawn_fish_ids, keys = db_data
            
            # Check data types
            if not isinstance(embeddings, torch.Tensor):
                issues.append("Embeddings is not a torch.Tensor")
            
            if not isinstance(internal_ids, list):
                issues.append("internal_ids is not a list")
            
            if not isinstance(image_ids, list):
                issues.append("image_ids is not a list")
            
            if not isinstance(annotation_ids, list):
                issues.append("annotation_ids is not a list")
            
            if not isinstance(drawn_fish_ids, list):
                issues.append("drawn_fish_ids is not a list")
            
            if not isinstance(keys, dict):
                issues.append("keys is not a dict")
            
            # Check length consistency
            lengths = [
                embeddings.shape[0] if isinstance(embeddings, torch.Tensor) else 0,
                len(internal_ids),
                len(image_ids),
                len(annotation_ids),
                len(drawn_fish_ids)
            ]
            
            if len(set(lengths)) > 1:
                issues.append(f"Inconsistent data lengths: {lengths}")
                recommendations.append("Truncate to minimum length or repair data")
            
            # Check embedding dimensions
            if isinstance(embeddings, torch.Tensor) and embeddings.numel() > 0:
                if len(embeddings.shape) != 2:
                    issues.append(f"Invalid embedding shape: {embeddings.shape}")
                elif embeddings.shape[1] != 128:
                    issues.append(f"Unexpected embedding dimension: {embeddings.shape[1]}, expected 128")
            
            # Check for duplicate IDs
            if len(image_ids) != len(set(image_ids)):
                issues.append("Duplicate image_ids found")
                recommendations.append("Remove duplicate entries")
            
            # Load and check labels
            labels_valid = True
            try:
                if self.labels_path.exists():
                    with open(self.labels_path, 'r', encoding='utf-8') as f:
                        labels = json.load(f)
                    
                    if not isinstance(labels, dict):
                        issues.append("Labels file is not a valid JSON dict")
                        labels_valid = False
                else:
                    issues.append("Labels file not found")
                    labels_valid = False
            except Exception as e:
                issues.append(f"Error reading labels file: {e}")
                labels_valid = False
            
            # Summary
            is_valid = len(issues) == 0
            
            result = {
                'valid': is_valid,
                'database_size': embeddings.shape[0] if isinstance(embeddings, torch.Tensor) else 0,
                'embedding_dimension': embeddings.shape[1] if isinstance(embeddings, torch.Tensor) and embeddings.numel() > 0 else 'N/A',
                'labels_valid': labels_valid,
                'issues': issues,
                'recommendations': recommendations,
                'data_lengths': {
                    'embeddings': lengths[0],
                    'internal_ids': lengths[1],
                    'image_ids': lengths[2],
                    'annotation_ids': lengths[3],
                    'drawn_fish_ids': lengths[4]
                }
            }
            
            if is_valid:
                logger.info("✅ Database integrity check passed")
            else:
                logger.warning(f"⚠️ Database integrity issues found: {len(issues)} issues")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error during integrity check: {e}")
            return {
                'valid': False,
                'error': str(e),
                'issues': [f"Exception during validation: {e}"],
                'recommendations': ['Check file permissions and format']
            }
    
    def repair_database(self) -> bool:
        """Perbaiki database yang rusak"""
        logger.info("🔧 Attempting to repair database...")
        
        try:
            integrity_check = self.validate_database_integrity()
            
            if integrity_check['valid']:
                logger.info("✅ Database is already valid, no repair needed")
                return True
            
            # Backup before repair
            self.create_backup("before_repair")
            
            # Load database
            db_data = torch.load(self.db_path)
            
            if len(db_data) != 6:
                logger.error("❌ Cannot repair: invalid structure")
                return False
            
            embeddings, internal_ids, image_ids, annotation_ids, drawn_fish_ids, keys = db_data
            
            # Fix length inconsistencies
            lengths = [
                embeddings.shape[0] if isinstance(embeddings, torch.Tensor) else 0,
                len(internal_ids),
                len(image_ids),
                len(annotation_ids),
                len(drawn_fish_ids)
            ]
            
            min_length = min(lengths)
            logger.info(f"Truncating all data to minimum length: {min_length}")
            
            # Truncate to minimum length
            if isinstance(embeddings, torch.Tensor) and embeddings.shape[0] > min_length:
                embeddings = embeddings[:min_length]
            
            internal_ids = internal_ids[:min_length]
            image_ids = image_ids[:min_length]
            annotation_ids = annotation_ids[:min_length]
            drawn_fish_ids = drawn_fish_ids[:min_length]
            
            # Remove duplicates from image_ids while maintaining order
            seen_ids = set()
            unique_indices = []
            
            for i, img_id in enumerate(image_ids):
                if img_id not in seen_ids:
                    seen_ids.add(img_id)
                    unique_indices.append(i)
            
            if len(unique_indices) < len(image_ids):
                logger.info(f"Removing {len(image_ids) - len(unique_indices)} duplicate entries")
                
                # Keep only unique entries
                embeddings = embeddings[unique_indices]
                internal_ids = [internal_ids[i] for i in unique_indices]
                image_ids = [image_ids[i] for i in unique_indices]
                annotation_ids = [annotation_ids[i] for i in unique_indices]
                drawn_fish_ids = [drawn_fish_ids[i] for i in unique_indices]
            
            # Save repaired database
            repaired_data = [embeddings, internal_ids, image_ids, annotation_ids, drawn_fish_ids, keys]
            torch.save(repaired_data, self.db_path)
            
            logger.info("✅ Database repair completed")
            
            # Validate repair
            post_repair_check = self.validate_database_integrity()
            if post_repair_check['valid']:
                logger.info("✅ Repair successful - database is now valid")
                return True
            else:
                logger.warning("⚠️ Repair attempted but issues remain")
                return False
            
        except Exception as e:
            logger.error(f"❌ Error during repair: {e}")
            return False
    
    def create_backup(self, suffix: str = "") -> str:
        """Buat backup database dengan timestamp"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"database_backup_{timestamp}"
            if suffix:
                backup_name += f"_{suffix}"
            backup_name += ".pt"
            
            backup_path = self.backup_dir / backup_name
            
            if self.db_path.exists():
                shutil.copy2(self.db_path, backup_path)
                logger.info(f"✅ Backup created: {backup_path}")
                return str(backup_path)
            else:
                logger.warning("⚠️ Database file not found, no backup created")
                return ""
        except Exception as e:
            logger.error(f"❌ Error creating backup: {e}")
            return ""
    
    def restore_from_backup(self, backup_path: str) -> bool:
        """Restore database dari backup"""
        try:
            backup_file = Path(backup_path)
            
            if not backup_file.exists():
                logger.error(f"❌ Backup file not found: {backup_path}")
                return False
            
            # Create current backup before restore
            self.create_backup("before_restore")
            
            # Restore
            shutil.copy2(backup_file, self.db_path)
            
            # Validate restored database
            integrity_check = self.validate_database_integrity()
            
            if integrity_check['valid']:
                logger.info(f"✅ Database restored successfully from {backup_path}")
                return True
            else:
                logger.warning(f"⚠️ Database restored but has integrity issues")
                return False
            
        except Exception as e:
            logger.error(f"❌ Error restoring backup: {e}")
            return False
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """List semua backup yang tersedia"""
        try:
            backups = []
            
            for backup_file in self.backup_dir.glob("database_backup_*.pt"):
                stat = backup_file.stat()
                backups.append({
                    'filename': backup_file.name,
                    'path': str(backup_file),
                    'size_mb': stat.st_size / (1024 * 1024),
                    'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
            
            # Sort by creation time (newest first)
            backups.sort(key=lambda x: x['created'], reverse=True)
            
            logger.info(f"📋 Found {len(backups)} backup(s)")
            return backups
            
        except Exception as e:
            logger.error(f"❌ Error listing backups: {e}")
            return []
    
    def clean_old_backups(self, keep_count: int = 10) -> int:
        """Hapus backup lama, keep hanya sejumlah backup terbaru"""
        try:
            backups = self.list_backups()
            
            if len(backups) <= keep_count:
                logger.info(f"✅ Only {len(backups)} backups found, no cleanup needed")
                return 0
            
            # Hapus backup lama
            deleted_count = 0
            for backup in backups[keep_count:]:
                try:
                    Path(backup['path']).unlink()
                    deleted_count += 1
                    logger.info(f"🗑️ Deleted old backup: {backup['filename']}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to delete {backup['filename']}: {e}")
            
            logger.info(f"✅ Cleanup completed: {deleted_count} old backups deleted")
            return deleted_count
            
        except Exception as e:
            logger.error(f"❌ Error during backup cleanup: {e}")
            return 0
    
    def export_database_info(self, output_path: str) -> bool:
        """Export informasi database ke file JSON"""
        try:
            # Get database stats
            integrity_check = self.validate_database_integrity()
            
            # Get additional info
            db_data = torch.load(self.db_path)
            embeddings, internal_ids, image_ids, annotation_ids, drawn_fish_ids, keys = db_data
            
            # Load labels
            labels = {}
            if self.labels_path.exists():
                with open(self.labels_path, 'r', encoding='utf-8') as f:
                    labels = json.load(f)
            
            # Compile export data
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'database_path': str(self.db_path),
                'labels_path': str(self.labels_path),
                'integrity_check': integrity_check,
                'statistics': {
                    'total_embeddings': embeddings.shape[0] if isinstance(embeddings, torch.Tensor) else 0,
                    'embedding_dimension': embeddings.shape[1] if isinstance(embeddings, torch.Tensor) and embeddings.numel() > 0 else None,
                    'total_species': len(labels),
                    'database_size_mb': self.db_path.stat().st_size / (1024 * 1024) if self.db_path.exists() else 0
                },
                'species_list': labels,
                'sample_data': {
                    'first_10_image_ids': image_ids[:10] if len(image_ids) >= 10 else image_ids,
                    'keys_preview': dict(list(keys.items())[:5]) if isinstance(keys, dict) else {}
                }
            }
            
            # Save export
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Database info exported to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error exporting database info: {e}")
            return False

def main():
    """Demo penggunaan Database Utilities"""
    print("🔧 Database Utilities Demo")
    print("=" * 50)
    
    # Initialize utilities
    db_path = "../models/classification/database.pt"
    labels_path = "../models/classification/labels.json"
    
    utils = DatabaseUtilities(db_path, labels_path)
    
    # 1. Validate database integrity
    print("\n🔍 Database Integrity Check:")
    integrity = utils.validate_database_integrity()
    
    print(f"   Valid: {integrity['valid']}")
    if 'database_size' in integrity:
        print(f"   Database size: {integrity['database_size']} entries")
    if 'embedding_dimension' in integrity:
        print(f"   Embedding dimension: {integrity['embedding_dimension']}")
    
    if integrity['issues']:
        print("   Issues found:")
        for issue in integrity['issues']:
            print(f"     - {issue}")
    
    if integrity['recommendations']:
        print("   Recommendations:")
        for rec in integrity['recommendations']:
            print(f"     - {rec}")
    
    # 2. List backups
    print("\n📋 Available Backups:")
    backups = utils.list_backups()
    
    if backups:
        for i, backup in enumerate(backups[:5]):  # Show first 5
            print(f"   {i+1}. {backup['filename']}")
            print(f"      Size: {backup['size_mb']:.2f} MB")
            print(f"      Created: {backup['created']}")
    else:
        print("   No backups found")
    
    # 3. Create backup
    print("\n💾 Creating Backup:")
    backup_path = utils.create_backup("demo")
    if backup_path:
        print(f"   ✅ Backup created: {backup_path}")
    else:
        print("   ❌ Failed to create backup")
    
    # 4. Export database info
    print("\n📤 Exporting Database Info:")
    export_path = "database_info_export.json"
    success = utils.export_database_info(export_path)
    if success:
        print(f"   ✅ Database info exported: {export_path}")
    else:
        print("   ❌ Failed to export database info")
    
    # 5. Clean old backups (dry run)
    print("\n🧹 Backup Cleanup (keeping 5 most recent):")
    deleted_count = utils.clean_old_backups(keep_count=5)
    print(f"   🗑️ Deleted {deleted_count} old backup(s)")
    
    print("\n✅ Database Utilities Demo Completed!")

if __name__ == "__main__":
    main()