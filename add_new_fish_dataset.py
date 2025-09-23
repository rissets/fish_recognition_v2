# -*- coding: utf-8 -*-
"""
Script untuk menambahkan dataset ikan baru ke dalam sistem fish recognition
menggunakan vector embedding tanpa melatih ulang model detection dan segmentation.
"""

import os
import sys
import json
import cv2
import numpy as np
import torch
import logging
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import glob
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import model inference classes
from models.classification.inference import EmbeddingClassifier
from models.detection.inference import YOLOInference
from models.segmentation.inference import Inference

class FishDatasetManager:
    def __init__(self, model_dirs):
        """
        Initialize Fish Dataset Manager dengan model yang sudah ada
        """
        self.model_dirs = model_dirs
        self.load_models()
        self.load_existing_labels()
        self.load_existing_database()
        
    def load_models(self):
        """Load semua model yang diperlukan"""
        logging.info("Loading models...")
        
        # Classification model
        self.classifier = EmbeddingClassifier(
            os.path.join(self.model_dirs['classification'], 'model.ts'),
            os.path.join(self.model_dirs['classification'], 'database.pt')
        )
        
        # Detection model
        self.detector = YOLOInference(
            os.path.join(self.model_dirs['detection'], 'model.ts'),
            imsz=(640, 640),
            conf_threshold=0.9,
            nms_threshold=0.3,
            yolo_ver='v10'
        )
        
        # Segmentation model
        self.segmentator = Inference(
            model_path=os.path.join(self.model_dirs['segmentation'], 'model.ts'),
            image_size=416
        )
        
        logging.info("All models loaded successfully!")
        
    def load_existing_labels(self):
        """Load label yang sudah ada"""
        labels_path = os.path.join(self.model_dirs['classification'], 'labels.json')
        try:
            with open(labels_path, 'r', encoding='utf-8') as f:
                self.existing_labels = json.load(f)
            logging.info(f"Loaded {len(self.existing_labels)} existing labels")
        except FileNotFoundError:
            self.existing_labels = {}
            logging.warning("No existing labels found, starting fresh")
            
    def load_existing_database(self):
        """Load database embedding yang sudah ada"""
        database_path = os.path.join(self.model_dirs['classification'], 'database.pt')
        try:
            # Format asli database: [embeddings, internal_ids, image_ids, annotation_ids, drawn_fish_ids, keys]
            database_data = torch.load(database_path)
            if len(database_data) == 6:
                self.existing_database = {
                    'embeddings': database_data[0],      # tensor embeddings
                    'internal_ids': database_data[1],    # list species IDs
                    'image_ids': database_data[2],       # list image IDs
                    'annotation_ids': database_data[3],  # list annotation IDs  
                    'drawn_fish_ids': database_data[4],  # list drawn fish IDs
                    'keys': database_data[5]             # dict with species info
                }
                logging.info("Loaded existing embedding database with proper structure")
            else:
                raise ValueError("Unexpected database format")
        except (FileNotFoundError, ValueError) as e:
            self.existing_database = {
                'embeddings': torch.empty(0, 512),  # Assuming 512-dim embeddings
                'internal_ids': [],
                'image_ids': [],
                'annotation_ids': [],
                'drawn_fish_ids': [],
                'keys': {}
            }
            logging.warning(f"No existing database found or invalid format, starting fresh: {e}")
    
    def extract_fish_features(self, image_path):
        """
        Extract fish dari image menggunakan detection dan segmentation
        """
        # Load image
        img_bgr = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Detect fish
        boxes = self.detector.predict(img_rgb)[0]
        
        extracted_features = []
        
        for i, box in enumerate(boxes):
            try:
                # Get cropped fish
                cropped_fish_bgr = box.get_mask_BGR()
                cropped_fish_rgb = box.get_mask_RGB()
                
                # Segmentation untuk mendapatkan mask yang lebih presisi
                segmented_polygons = self.segmentator.predict(cropped_fish_bgr)[0]
                fish_mask = segmented_polygons.mask_polygon(cropped_fish_rgb)
                
                # Extract embedding menggunakan classification model
                with torch.no_grad():
                    # Preprocess image untuk model classification
                    processed_img = self._preprocess_for_classification(cropped_fish_bgr)
                    
                    # Model returns embedding and fc_output
                    embedding, fc_output = self.classifier.model(processed_img)
                    
                extracted_features.append({
                    'embedding': embedding[0].cpu().numpy(),  # Take first element from batch
                    'cropped_image': cropped_fish_bgr,
                    'mask': fish_mask,
                    'bbox': [box.x1, box.y1, box.x2, box.y2],
                    'segmentation': segmented_polygons
                })
                
            except Exception as e:
                logging.error(f"Error processing fish {i} in {image_path}: {e}")
                continue
                
        return extracted_features
    
    def _preprocess_for_classification(self, image_bgr):
        """Preprocess image untuk classification model"""
        from PIL import Image
        from torchvision import transforms
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)
        
        # Use the same preprocessing as classifier
        loader = transforms.Compose([
            transforms.Resize((224, 224), Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Process and add batch dimension
        processed_tensor = loader(pil_image).unsqueeze(0)
        
        return processed_tensor
    
    def add_new_fish_species(self, dataset_path, species_name, species_id=None):
        """
        Menambahkan spesies ikan baru ke dalam database
        """
        if species_id is None:
            # Generate new species ID
            existing_ids = [int(k) for k in self.existing_labels.keys() if k.isdigit()]
            species_id = max(existing_ids) + 1 if existing_ids else 0
        
        species_id = str(species_id)
        
        # Add to labels
        self.existing_labels[species_id] = species_name
        
        # Process all images in dataset path
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(dataset_path, ext)))
            image_paths.extend(glob.glob(os.path.join(dataset_path, ext.upper())))
        
        logging.info(f"Found {len(image_paths)} images for species {species_name}")
        
        new_embeddings = []
        processed_count = 0
        
        for img_path in image_paths:
            logging.info(f"Processing {img_path}")
            
            features = self.extract_fish_features(img_path)
            
            for feature in features:
                new_embeddings.append(feature['embedding'])
                processed_count += 1
        
        if new_embeddings:
            # Add to existing database with proper structure
            embeddings_tensor = self.existing_database['embeddings']
            internal_ids = list(self.existing_database['internal_ids'])
            image_ids = list(self.existing_database['image_ids'])
            annotation_ids = list(self.existing_database['annotation_ids'])
            drawn_fish_ids = list(self.existing_database['drawn_fish_ids'])
            keys = dict(self.existing_database['keys'])
            
            # Convert new embeddings to tensor
            new_emb_array = np.array(new_embeddings)
            if len(new_emb_array.shape) == 3:  # Remove batch dimension if exists
                new_emb_array = new_emb_array.squeeze(1)
            new_emb_tensor = torch.from_numpy(new_emb_array)
            
            # Concatenate embeddings
            updated_embeddings = torch.cat([embeddings_tensor, new_emb_tensor], dim=0)
            
            # Generate unique IDs for new entries
            import uuid
            
            # Add new metadata for each embedding
            for i in range(len(new_embeddings)):
                internal_ids.append(int(species_id))
                image_ids.append(str(uuid.uuid4()))  # Generate UUID string
                annotation_ids.append(str(uuid.uuid4()))  # Generate UUID string
                drawn_fish_ids.append(str(uuid.uuid4()))  # Generate UUID string
            
            # Add species info to keys if not exists
            if int(species_id) not in keys:
                keys[int(species_id)] = {
                    'label': species_name,
                    'species_id': int(species_id)
                }
            
            # Update database structure
            self.existing_database = {
                'embeddings': updated_embeddings,
                'internal_ids': internal_ids,
                'image_ids': image_ids,
                'annotation_ids': annotation_ids,
                'drawn_fish_ids': drawn_fish_ids,
                'keys': keys
            }
            
            logging.info(f"Added {len(new_embeddings)} embeddings for species {species_name}")
        
        return processed_count
    
    def save_updated_database(self):
        """Save updated database dan labels"""
        # Save labels
        labels_path = os.path.join(self.model_dirs['classification'], 'labels.json')
        with open(labels_path, 'w', encoding='utf-8') as f:
            json.dump(self.existing_labels, f, ensure_ascii=False, indent=2)
        
        # Convert database back to original format
        database_data = [
            self.existing_database['embeddings'],      # tensor
            self.existing_database['internal_ids'],    # list
            self.existing_database['image_ids'],       # list  
            self.existing_database['annotation_ids'],  # list
            self.existing_database['drawn_fish_ids'],  # list
            self.existing_database['keys']             # dict
        ]
        
        # Save database
        database_path = os.path.join(self.model_dirs['classification'], 'database.pt')
        torch.save(database_data, database_path)
        
        # Backup original database
        backup_path = database_path + '.backup'
        if os.path.exists(backup_path):
            os.remove(backup_path)
        
        logging.info("Database and labels saved successfully!")
    
    def predict_fish(self, image_path, top_k=5):
        """
        Prediksi ikan pada image baru dengan detection, classification, dan segmentation
        """
        # Load image
        img_bgr = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        visualize_img = img_rgb.copy()
        
        # Detect fish
        boxes = self.detector.predict(img_rgb)[0]
        
        results = []
        
        for i, box in enumerate(boxes):
            try:
                # Get cropped fish
                cropped_fish_bgr = box.get_mask_BGR()
                cropped_fish_rgb = box.get_mask_RGB()
                
                # Segmentation
                segmented_polygons = self.segmentator.predict(cropped_fish_bgr)[0]
                fish_mask = segmented_polygons.mask_polygon(cropped_fish_rgb)
                
                # Move segmentation to original image coordinates
                segmented_polygons.move_to(box.x1, box.y1)
                segmented_polygons.draw_polygon(visualize_img)
                
                # Classification menggunakan database yang sudah diupdate
                classification_result = self.classifier.batch_inference([cropped_fish_bgr])[0]
                
                # Draw results
                if classification_result:
                    label = f"{classification_result[0]['name']} | {classification_result[0]['accuracy']:.3f}"
                    box.draw_label(visualize_img, label)
                else:
                    box.draw_label(visualize_img, "Unknown")
                
                box.draw_box(visualize_img)
                
                results.append({
                    'fish_id': i,
                    'classification': classification_result,
                    'bbox': [box.x1, box.y1, box.x2, box.y2],
                    'segmentation_mask': fish_mask,
                    'cropped_image': cropped_fish_rgb
                })
                
            except Exception as e:
                logging.error(f"Error processing fish {i}: {e}")
                continue
        
        return results, visualize_img
    
    def display_results(self, results, visualize_img):
        """Display hasil prediksi"""
        plt.figure(figsize=(15, 10))
        
        # Main image with detections
        plt.subplot(2, len(results) + 1, 1)
        plt.imshow(visualize_img)
        plt.title("Detection + Segmentation Results")
        plt.axis('off')
        
        # Individual fish results
        for i, result in enumerate(results):
            # Cropped fish
            plt.subplot(2, len(results) + 1, i + 2)
            plt.imshow(result['cropped_image'])
            plt.title(f"Fish {result['fish_id']}")
            plt.axis('off')
            
            # Segmentation mask
            plt.subplot(2, len(results) + 1, len(results) + i + 2)
            plt.imshow(result['segmentation_mask'])
            plt.title(f"Segmentation {result['fish_id']}")
            plt.axis('off')
            
            # Print classification results
            print(f"\nFish {result['fish_id']} Classification Results:")
            print("-" * 40)
            for j, fish_info in enumerate(result['classification'][:3], 1):
                print(f"{j}. {fish_info['name']}")
                print(f"   Species ID: {fish_info['species_id']}")
                print(f"   Accuracy: {fish_info['accuracy']:.3f}")
                print(f"   Distance: {fish_info['distance']:.3f}")
        
        plt.tight_layout()
        plt.show()

def main():
    """Main function untuk menjalankan script"""
    
    # Model directories
    MODEL_DIRS = {
        'classification': "models/classification",
        'segmentation': "models/segmentation", 
        'detection': "models/detection"
    }
    
    # Initialize dataset manager
    fish_manager = FishDatasetManager(MODEL_DIRS)
    
    # Contoh penggunaan:
    # 1. Menambahkan spesies baru
    print("=== Menambahkan Dataset Baru ===")
    
    # Ganti path ini dengan path ke folder dataset ikan baru Anda
    new_fish_dataset_path = "dataset/ikan_db_v1/images/mujair"  # Contoh
    new_species_name = "Mujair"
    
    # Cek apakah path exists
    if os.path.exists(new_fish_dataset_path):
        print(f"Adding new species: {new_species_name}")
        processed_count = fish_manager.add_new_fish_species(
            new_fish_dataset_path, 
            new_species_name
        )
        print(f"Processed {processed_count} fish images")
        
        # Save updated database
        fish_manager.save_updated_database()
        print("Database updated successfully!")
    else:
        print(f"Dataset path {new_fish_dataset_path} not found!")
        print("Available species in dataset:")
        dataset_path = "dataset/ikan_db_v1/images"
        if os.path.exists(dataset_path):
            species_folders = [f for f in os.listdir(dataset_path) 
                             if os.path.isdir(os.path.join(dataset_path, f))]
            for species in species_folders[:10]:  # Show first 10
                print(f"  - {species}")
            print(f"... and {len(species_folders)-10} more species")
    
    # 2. Test prediksi pada image baru
    print("\n=== Testing Prediksi ===")
    test_image_path = "images/mujair1.jpg"  # Ganti dengan path image test Anda
    
    if os.path.exists(test_image_path):
        print(f"Testing prediction on: {test_image_path}")
        results, visualize_img = fish_manager.predict_fish(test_image_path)
        
        if results:
            fish_manager.display_results(results, visualize_img)
        else:
            print("No fish detected in the image!")
    else:
        print(f"Test image {test_image_path} not found!")
        # List available test images
        test_dir = "images"
        if os.path.exists(test_dir):
            test_files = [f for f in os.listdir(test_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print("Available test images:")
            for img in test_files:
                print(f"  - {img}")

if __name__ == "__main__":
    main()