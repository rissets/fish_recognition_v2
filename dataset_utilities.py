# -*- coding: utf-8 -*-
"""
Utility functions untuk dataset management dan batch processing
"""

import os
import json
import cv2
import numpy as np
import logging
from pathlib import Path
import glob
import shutil
from collections import defaultdict

class DatasetUtilities:
    """Utility class untuk manajemen dataset"""
    
    @staticmethod
    def list_available_species(dataset_base_path="dataset/ikan_db_v1/images"):
        """List semua spesies yang tersedia dalam dataset"""
        if not os.path.exists(dataset_base_path):
            print(f"Dataset path {dataset_base_path} tidak ditemukan!")
            return []
        
        species_folders = [f for f in os.listdir(dataset_base_path) 
                          if os.path.isdir(os.path.join(dataset_base_path, f))]
        
        species_info = []
        for species in species_folders:
            species_path = os.path.join(dataset_base_path, species)
            image_count = len(glob.glob(os.path.join(species_path, "*.jpg")) + 
                            glob.glob(os.path.join(species_path, "*.png")) +
                            glob.glob(os.path.join(species_path, "*.jpeg")))
            species_info.append({
                'name': species,
                'path': species_path,
                'image_count': image_count
            })
        
        # Sort by image count
        species_info.sort(key=lambda x: x['image_count'], reverse=True)
        return species_info
    
    @staticmethod
    def display_species_info(species_info, max_display=20):
        """Display informasi spesies"""
        print(f"\nDitemukan {len(species_info)} spesies ikan:")
        print("-" * 60)
        print(f"{'No.':<4} {'Species Name':<30} {'Images':<10}")
        print("-" * 60)
        
        for i, info in enumerate(species_info[:max_display], 1):
            print(f"{i:<4} {info['name']:<30} {info['image_count']:<10}")
        
        if len(species_info) > max_display:
            print(f"... dan {len(species_info) - max_display} spesies lainnya")
    
    @staticmethod
    def check_current_labels(labels_path="models/classification/labels.json"):
        """Check label yang sudah ada"""
        try:
            with open(labels_path, 'r', encoding='utf-8') as f:
                labels = json.load(f)
            
            print(f"\nLabel yang sudah ada ({len(labels)} spesies):")
            print("-" * 40)
            for species_id, name in labels.items():
                print(f"ID {species_id}: {name}")
            
            return labels
        except FileNotFoundError:
            print(f"File label {labels_path} tidak ditemukan!")
            return {}
    
    @staticmethod
    def backup_current_database(model_dir="models/classification"):
        """Backup database dan labels saat ini"""
        import datetime
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = os.path.join(model_dir, f"backup_{timestamp}")
        os.makedirs(backup_dir, exist_ok=True)
        
        # Backup files
        files_to_backup = ['database.pt', 'labels.json']
        for file in files_to_backup:
            src = os.path.join(model_dir, file)
            if os.path.exists(src):
                dst = os.path.join(backup_dir, file)
                shutil.copy2(src, dst)
                print(f"Backed up {file} to {backup_dir}")
        
        return backup_dir
    
    @staticmethod
    def validate_image_files(dataset_path):
        """Validate image files dalam dataset"""
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        valid_images = []
        invalid_images = []
        
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_paths = glob.glob(os.path.join(dataset_path, ext))
            image_paths.extend(glob.glob(os.path.join(dataset_path, ext.upper())))
            
            for img_path in image_paths:
                try:
                    # Try to read image
                    img = cv2.imread(img_path)
                    if img is not None and img.shape[0] > 0 and img.shape[1] > 0:
                        valid_images.append(img_path)
                    else:
                        invalid_images.append(img_path)
                except Exception as e:
                    invalid_images.append(img_path)
                    logging.error(f"Error reading {img_path}: {e}")
        
        return valid_images, invalid_images

class BatchProcessor:
    """Class untuk batch processing multiple species"""
    
    def __init__(self, fish_manager):
        self.fish_manager = fish_manager
        
    def add_multiple_species(self, species_config):
        """
        Add multiple species sekaligus
        species_config: list of dict dengan format:
        [
            {'path': 'path/to/species1', 'name': 'Species1'},
            {'path': 'path/to/species2', 'name': 'Species2'},
        ]
        """
        results = []
        
        # Backup current database
        backup_dir = DatasetUtilities.backup_current_database()
        print(f"Database backed up to: {backup_dir}")
        
        for i, config in enumerate(species_config):
            try:
                print(f"\nProcessing {i+1}/{len(species_config)}: {config['name']}")
                print("-" * 50)
                
                # Validate dataset path
                if not os.path.exists(config['path']):
                    print(f"Path {config['path']} tidak ditemukan, skip...")
                    results.append({'species': config['name'], 'status': 'failed', 'reason': 'path not found'})
                    continue
                
                # Validate images
                valid_images, invalid_images = DatasetUtilities.validate_image_files(config['path'])
                
                if not valid_images:
                    print(f"Tidak ada image valid di {config['path']}, skip...")
                    results.append({'species': config['name'], 'status': 'failed', 'reason': 'no valid images'})
                    continue
                
                if invalid_images:
                    print(f"Warning: {len(invalid_images)} invalid images akan di-skip")
                
                # Process species
                processed_count = self.fish_manager.add_new_fish_species(
                    config['path'], 
                    config['name']
                )
                
                results.append({
                    'species': config['name'], 
                    'status': 'success', 
                    'processed_count': processed_count,
                    'valid_images': len(valid_images),
                    'invalid_images': len(invalid_images)
                })
                
                print(f"Successfully processed {processed_count} embeddings for {config['name']}")
                
            except Exception as e:
                logging.error(f"Error processing {config['name']}: {e}")
                results.append({'species': config['name'], 'status': 'failed', 'reason': str(e)})
        
        # Save updated database
        try:
            self.fish_manager.save_updated_database()
            print("\n" + "="*50)
            print("BATCH PROCESSING COMPLETED!")
            print("="*50)
        except Exception as e:
            print(f"Error saving database: {e}")
            print(f"You can restore from backup: {backup_dir}")
        
        return results
    
    def display_batch_results(self, results):
        """Display hasil batch processing"""
        successful = [r for r in results if r['status'] == 'success']
        failed = [r for r in results if r['status'] == 'failed']
        
        print(f"\nBATCH PROCESSING SUMMARY:")
        print(f"Total species: {len(results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        
        if successful:
            print(f"\nSUCCESSFUL SPECIES:")
            total_embeddings = 0
            for result in successful:
                print(f"  - {result['species']}: {result['processed_count']} embeddings")
                total_embeddings += result['processed_count']
            print(f"Total new embeddings: {total_embeddings}")
        
        if failed:
            print(f"\nFAILED SPECIES:")
            for result in failed:
                print(f"  - {result['species']}: {result['reason']}")

def interactive_species_selector():
    """Interactive function untuk memilih spesies yang akan ditambahkan"""
    print("=== INTERACTIVE SPECIES SELECTOR ===")
    
    # Get available species
    species_info = DatasetUtilities.list_available_species()
    
    if not species_info:
        print("Tidak ada dataset yang ditemukan!")
        return []
    
    # Display species
    DatasetUtilities.display_species_info(species_info)
    
    # Get current labels
    current_labels = DatasetUtilities.check_current_labels()
    current_species_names = set(current_labels.values())
    
    # Filter species yang belum ada
    new_species = [s for s in species_info if s['name'] not in current_species_names]
    
    if not new_species:
        print("\nSemua spesies sudah ada dalam database!")
        return []
    
    print(f"\nSpesies baru yang dapat ditambahkan ({len(new_species)}):")
    print("-" * 60)
    for i, info in enumerate(new_species, 1):
        print(f"{i:<4} {info['name']:<30} {info['image_count']:<10}")
    
    # Interactive selection
    selected_species = []
    print(f"\nPilih spesies yang ingin ditambahkan (contoh: 1,3,5 atau 'all' untuk semua):")
    print("Atau ketik 'top10' untuk 10 spesies dengan image terbanyak")
    
    choice = input("Pilihan Anda: ").strip()
    
    if choice.lower() == 'all':
        selected_species = new_species
    elif choice.lower() == 'top10':
        selected_species = new_species[:10]
    else:
        try:
            indices = [int(x.strip()) - 1 for x in choice.split(',')]
            selected_species = [new_species[i] for i in indices if 0 <= i < len(new_species)]
        except (ValueError, IndexError):
            print("Input tidak valid!")
            return []
    
    # Confirm selection
    if selected_species:
        print(f"\nAnda memilih {len(selected_species)} spesies:")
        for species in selected_species:
            print(f"  - {species['name']} ({species['image_count']} images)")
        
        confirm = input("\nLanjutkan? (y/n): ").strip().lower()
        if confirm == 'y':
            return [{'path': s['path'], 'name': s['name']} for s in selected_species]
    
    return []

def main():
    """Main function untuk utilities"""
    print("=== FISH DATASET UTILITIES ===")
    
    # Show available options
    print("\nOpsi yang tersedia:")
    print("1. List semua spesies yang tersedia")
    print("2. Check label yang sudah ada")
    print("3. Interactive species selector")
    print("4. Validate dataset images")
    
    choice = input("\nPilih opsi (1-4): ").strip()
    
    if choice == '1':
        species_info = DatasetUtilities.list_available_species()
        DatasetUtilities.display_species_info(species_info)
        
    elif choice == '2':
        DatasetUtilities.check_current_labels()
        
    elif choice == '3':
        selected_config = interactive_species_selector()
        if selected_config:
            print(f"\nSelected species config:")
            for config in selected_config:
                print(f"  - {config['name']}: {config['path']}")
        else:
            print("Tidak ada spesies yang dipilih.")
            
    elif choice == '4':
        dataset_path = input("Masukkan path dataset: ").strip()
        if os.path.exists(dataset_path):
            valid, invalid = DatasetUtilities.validate_image_files(dataset_path)
            print(f"Valid images: {len(valid)}")
            print(f"Invalid images: {len(invalid)}")
            if invalid:
                print("Invalid files:")
                for img in invalid:
                    print(f"  - {img}")
        else:
            print("Path tidak ditemukan!")

if __name__ == "__main__":
    main()