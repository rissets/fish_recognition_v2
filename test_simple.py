#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple test script untuk menambahkan satu spesies ikan
"""

import sys
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from add_new_fish_dataset import FishDatasetManager

def test_single_species():
    """Test menambahkan satu spesies"""
    
    MODEL_DIRS = {
        'classification': "models/classification",
        'segmentation': "models/segmentation", 
        'detection': "models/detection"
    }
    
    # Check models exist
    for model_type, model_dir in MODEL_DIRS.items():
        if not os.path.exists(model_dir):
            print(f"Model directory {model_dir} not found!")
            return
            
    # Initialize fish manager
    try:
        print("Loading models...")
        fish_manager = FishDatasetManager(MODEL_DIRS)
        print("Models loaded successfully!")
    except Exception as e:
        print(f"Error loading models: {e}")
        return
    
    # Test dengan spesies bandeng yang punya 1 image saja
    species_path = "dataset/ikan_db_v1/images/bandeng"
    species_name = "Bandeng"
    
    if not os.path.exists(species_path):
        print(f"Species path {species_path} not found!")
        print("Looking for alternatives...")
        
        # Cari spesies dengan sedikit images
        import glob
        dataset_base = "dataset/ikan_db_v1/images"
        alternatives = []
        
        for folder in os.listdir(dataset_base):
            folder_path = os.path.join(dataset_base, folder)
            if os.path.isdir(folder_path):
                image_count = len(glob.glob(os.path.join(folder_path, "*.jpg")))
                if 1 <= image_count <= 3:  # Small dataset
                    alternatives.append((folder, image_count))
        
        if alternatives:
            # Use first alternative
            species_name = alternatives[0][0]
            species_path = os.path.join(dataset_base, species_name)
            print(f"Using alternative: {species_name} with {alternatives[0][1]} images")
        else:
            print("No suitable alternative found!")
            return
    
    print(f"\nTesting with species: {species_name}")
    print(f"Path: {species_path}")
    
    # Check current labels
    print(f"Current labels count: {len(fish_manager.existing_labels)}")
    
    # Check if species already exists
    existing_names = set(fish_manager.existing_labels.values())
    if species_name in existing_names:
        print(f"Species {species_name} already exists in database!")
        return
    
    try:
        # Add species
        print("Adding species...")
        processed_count = fish_manager.add_new_fish_species(species_path, species_name)
        
        if processed_count > 0:
            print(f"Successfully processed {processed_count} embeddings")
            
            # Save database
            print("Saving database...")
            fish_manager.save_updated_database()
            print("Database saved successfully!")
            
            # Test prediction
            print("\nTesting prediction...")
            test_image = "images/mujair1.jpg"  # Use any test image
            if os.path.exists(test_image):
                results, viz_img = fish_manager.predict_fish(test_image)
                print(f"Detected {len(results)} fish in test image")
                for i, result in enumerate(results):
                    if result['classification']:
                        top_result = result['classification'][0]
                        print(f"Fish {i}: {top_result['name']} (accuracy: {top_result['accuracy']:.3f})")
            else:
                print(f"Test image {test_image} not found")
                
        else:
            print("No fish detected in the dataset images!")
            
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_single_species()