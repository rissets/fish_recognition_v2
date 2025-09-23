#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Demo Script untuk Fish Recognition dengan Dataset Baru
Script ini mendemonstrasikan cara menambahkan dataset baru dan melakukan prediksi
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Import custom modules
from add_new_fish_dataset import FishDatasetManager
from dataset_utilities import DatasetUtilities, BatchProcessor, interactive_species_selector

def setup_logging(verbose=False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def add_species_from_config(fish_manager, config_file):
    """Add species from JSON config file"""
    import json
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        species_list = config.get('species', [])
        if not species_list:
            print("No species found in config file!")
            return
        
        batch_processor = BatchProcessor(fish_manager)
        results = batch_processor.add_multiple_species(species_list)
        batch_processor.display_batch_results(results)
        
    except FileNotFoundError:
        print(f"Config file {config_file} not found!")
    except json.JSONDecodeError:
        print(f"Invalid JSON format in {config_file}")

def add_single_species(fish_manager, species_path, species_name):
    """Add single species"""
    if not os.path.exists(species_path):
        print(f"Species path {species_path} not found!")
        return
    
    print(f"Adding species: {species_name}")
    print(f"From path: {species_path}")
    
    # Validate images first
    valid_images, invalid_images = DatasetUtilities.validate_image_files(species_path)
    
    if not valid_images:
        print("No valid images found!")
        return
    
    print(f"Found {len(valid_images)} valid images")
    if invalid_images:
        print(f"Warning: {len(invalid_images)} invalid images will be skipped")
    
    # Backup current database
    backup_dir = DatasetUtilities.backup_current_database()
    print(f"Database backed up to: {backup_dir}")
    
    try:
        processed_count = fish_manager.add_new_fish_species(species_path, species_name)
        fish_manager.save_updated_database()
        
        print(f"Successfully added {processed_count} embeddings for {species_name}")
        print("Database updated!")
        
    except Exception as e:
        print(f"Error adding species: {e}")
        print(f"You can restore from backup: {backup_dir}")

def run_prediction_demo(fish_manager, test_dir):
    """Run prediction demo on test images"""
    if not os.path.exists(test_dir):
        print(f"Test directory {test_dir} not found!")
        return
    
    # Find test images
    import glob
    test_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        test_images.extend(glob.glob(os.path.join(test_dir, ext)))
        test_images.extend(glob.glob(os.path.join(test_dir, ext.upper())))
    
    if not test_images:
        print("No test images found!")
        return
    
    print(f"Found {len(test_images)} test images")
    
    for i, img_path in enumerate(test_images, 1):
        print(f"\n=== Testing Image {i}/{len(test_images)}: {os.path.basename(img_path)} ===")
        
        try:
            results, visualize_img = fish_manager.predict_fish(img_path)
            
            if results:
                print(f"Detected {len(results)} fish in the image")
                fish_manager.display_results(results, visualize_img)
            else:
                print("No fish detected in this image")
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

def interactive_mode():
    """Interactive mode untuk user-friendly operation"""
    print("=== FISH RECOGNITION INTERACTIVE MODE ===")
    print("\nInitializing models...")
    
    MODEL_DIRS = {
        'classification': "models/classification",
        'segmentation': "models/segmentation", 
        'detection': "models/detection"
    }
    
    # Check if models exist
    for model_type, model_dir in MODEL_DIRS.items():
        if not os.path.exists(model_dir):
            print(f"Model directory {model_dir} not found!")
            print("Please make sure all models are downloaded and extracted.")
            return
    
    try:
        fish_manager = FishDatasetManager(MODEL_DIRS)
        print("Models loaded successfully!")
    except Exception as e:
        print(f"Error loading models: {e}")
        return
    
    while True:
        print("\n" + "="*50)
        print("MAIN MENU")
        print("="*50)
        print("1. Show current database info")
        print("2. Add new species (interactive)")
        print("3. Add single species (manual)")
        print("4. Run prediction demo")
        print("5. List available species in dataset")
        print("6. Exit")
        
        choice = input("\nSelect option (1-6): ").strip()
        
        if choice == '1':
            # Show current database info
            labels = DatasetUtilities.check_current_labels()
            print(f"\nTotal species in database: {len(labels)}")
            
        elif choice == '2':
            # Interactive species addition
            selected_config = interactive_species_selector()
            if selected_config:
                batch_processor = BatchProcessor(fish_manager)
                results = batch_processor.add_multiple_species(selected_config)
                batch_processor.display_batch_results(results)
            
        elif choice == '3':
            # Manual single species addition
            species_path = input("Enter species dataset path: ").strip()
            species_name = input("Enter species name: ").strip()
            
            if species_path and species_name:
                add_single_species(fish_manager, species_path, species_name)
            else:
                print("Both path and name are required!")
                
        elif choice == '4':
            # Prediction demo
            test_dir = input("Enter test images directory [default: images]: ").strip()
            if not test_dir:
                test_dir = "images"
            run_prediction_demo(fish_manager, test_dir)
            
        elif choice == '5':
            # List available species
            species_info = DatasetUtilities.list_available_species()
            DatasetUtilities.display_species_info(species_info)
            
        elif choice == '6':
            print("Goodbye!")
            break
            
        else:
            print("Invalid choice! Please select 1-6.")

def main():
    """Main function dengan argument parsing"""
    parser = argparse.ArgumentParser(description='Fish Recognition Batch Demo')
    parser.add_argument('--mode', choices=['interactive', 'batch', 'predict'], 
                       default='interactive', help='Operation mode')
    parser.add_argument('--config', help='JSON config file for batch mode')
    parser.add_argument('--species-path', help='Path to species dataset')
    parser.add_argument('--species-name', help='Name of the species')
    parser.add_argument('--test-dir', default='images', help='Test images directory')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    MODEL_DIRS = {
        'classification': "models/classification",
        'segmentation': "models/segmentation", 
        'detection': "models/detection"
    }
    
    # Check models
    missing_models = []
    for model_type, model_dir in MODEL_DIRS.items():
        if not os.path.exists(model_dir):
            missing_models.append(model_dir)
    
    if missing_models:
        print("Missing model directories:")
        for model_dir in missing_models:
            print(f"  - {model_dir}")
        print("\nPlease run research_fishial.py first to download the models.")
        return
    
    if args.mode == 'interactive':
        interactive_mode()
        
    elif args.mode == 'batch':
        if not args.config:
            print("Config file required for batch mode!")
            return
            
        fish_manager = FishDatasetManager(MODEL_DIRS)
        add_species_from_config(fish_manager, args.config)
        
    elif args.mode == 'predict':
        fish_manager = FishDatasetManager(MODEL_DIRS)
        run_prediction_demo(fish_manager, args.test_dir)
        
    else:
        # Single species mode
        if not args.species_path or not args.species_name:
            print("Species path and name required!")
            return
            
        fish_manager = FishDatasetManager(MODEL_DIRS)
        add_single_species(fish_manager, args.species_path, args.species_name)

if __name__ == "__main__":
    main()