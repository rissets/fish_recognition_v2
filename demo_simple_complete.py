#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo sederhana untuk training dan prediction fish recognition
Menggunakan embedding approach untuk menambah spesies baru
"""

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
import uuid
from datetime import datetime

# Import model inference
from models.classification.inference import EmbeddingClassifier
from models.detection.inference import YOLOInference
from models.segmentation.inference import Inference

class SimpleFishSystem:
    def __init__(self):
        """Inisialisasi sistem recognition"""
        print("🐟 Initializing Fish Recognition System...")
        
        # Path model
        self.classifier_model_path = 'models/classification/model.ts'
        self.classifier_db_path = 'models/classification/database.pt'
        self.detector_model_path = 'models/detection/model.ts'
        self.segmentation_model_path = 'models/segmentation/model.ts'
        self.labels_path = 'models/classification/labels.json'
        
        # Inisialisasi model
        try:
            self.detector = YOLOInference(
                self.detector_model_path, 
                imsz=(640, 640), 
                conf_threshold=0.7, 
                nms_threshold=0.3, 
                yolo_ver='v10'
            )
            print("✅ Detector loaded")
            
            self.segmentator = Inference(
                model_path=self.segmentation_model_path, 
                image_size=416
            )
            print("✅ Segmentator loaded")
            
            self.classifier = EmbeddingClassifier(
                self.classifier_model_path, 
                self.classifier_db_path,
                labels_file=self.labels_path
            )
            print("✅ Classifier loaded")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise
        
        print("🎉 All models loaded successfully!")
    
    def show_database_info(self):
        """Menampilkan informasi database"""
        try:
            print("\n📊 Database Information:")
            
            # Load database format asli (tuple dengan 6 elemen)
            db_data = torch.load(self.classifier_db_path)
            embeddings = db_data[0]  # tensor embeddings
            internal_ids = db_data[1]  # list internal IDs
            image_ids = db_data[2]  # list image IDs
            
            print(f"   Total embeddings: {embeddings.shape[0] if embeddings.numel() > 0 else 0}")
            print(f"   Embedding dimension: {embeddings.shape[1] if embeddings.numel() > 0 else 'N/A'}")
            print(f"   Total image IDs: {len(image_ids)}")
            
            # Load labels
            if os.path.exists(self.labels_path):
                with open(self.labels_path, 'r', encoding='utf-8') as f:
                    labels = json.load(f)
                print(f"   Available species: {len(labels)}")
                print("   Species list:")
                for key, value in sorted(labels.items(), key=lambda x: int(x[0])):
                    print(f"     {key}: {value}")
            else:
                print("   Labels file not found")
                
        except Exception as e:
            print(f"❌ Error reading database: {e}")
    
    def add_new_fish_embedding(self, image_path, species_name):
        """
        Menambah embedding ikan baru ke database
        CATATAN: Ini adalah demo sederhana. Untuk production, perlu implementasi yang lebih robust.
        """
        print(f"\n🔄 Adding new fish: {species_name} from {image_path}")
        
        try:
            # Baca gambar
            if not os.path.exists(image_path):
                print(f"❌ Image not found: {image_path}")
                return False
            
            img_bgr = cv2.imread(image_path)
            if img_bgr is None:
                print(f"❌ Cannot read image: {image_path}")
                return False
            
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            # Detection
            print("   🔍 Detecting fish...")
            detection_results = self.detector.predict(img_rgb)
            if not detection_results or len(detection_results[0]) == 0:
                print("❌ No fish detected in image!")
                return False
            
            # Ambil deteksi terbaik
            box = detection_results[0][0]
            cropped_bgr = box.get_mask_BGR()
            print(f"   ✅ Fish detected with confidence: {box.score:.3f}")
            
            # Generate embedding
            print("   🧠 Generating embedding...")
            # Gunakan model langsung untuk mendapatkan embedding
            image = Image.fromarray(cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB))
            image_tensor = self.classifier.loader(image).unsqueeze(0).to('cpu')
            
            with torch.no_grad():
                embedding, _ = self.classifier.model(image_tensor)
                embedding = embedding[0]  # Ambil embedding pertama dari batch
            
            print(f"⚠️  CATATAN: Untuk demo ini, kita hanya menampilkan embedding.")
            print(f"   Embedding shape: {embedding.shape}")
            print(f"   Embedding norm: {torch.norm(embedding).item():.4f}")
            print(f"   Untuk menambah ke database secara permanen, diperlukan")
            print(f"   implementasi yang lebih kompleks yang mempertahankan struktur database.")
            
            return True
            
        except Exception as e:
            print(f"❌ Error processing image: {e}")
            return False
    
    def predict_fish_complete(self, image_path, show_visualization=True):
        """
        Prediksi lengkap: detection + classification + segmentation
        """
        print(f"\n🔮 Predicting fish in: {os.path.basename(image_path)}")
        
        try:
            # Baca gambar
            if not os.path.exists(image_path):
                print(f"❌ Image not found: {image_path}")
                return []
            
            img_bgr = cv2.imread(image_path)
            if img_bgr is None:
                print(f"❌ Cannot read image: {image_path}")
                return []
            
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            # 1. DETECTION
            print("   🔍 Step 1: Fish Detection...")
            detection_results = self.detector.predict(img_rgb)
            if not detection_results or len(detection_results[0]) == 0:
                print("❌ No fish detected!")
                return []
            
            print(f"✅ Detected {len(detection_results[0])} fish(es)")
            
            results = []
            
            for i, box in enumerate(detection_results[0]):
                print(f"\n   🐟 Processing Fish #{i+1}:")
                print(f"      Detection confidence: {box.score:.3f}")
                
                # Crop gambar
                cropped_bgr = box.get_mask_BGR()
                cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
                
                # 2. CLASSIFICATION
                print("   🧠 Step 2: Classification...")
                try:
                    # Cek apakah database memiliki data
                    db_data = torch.load(self.classifier_db_path)
                    embeddings = db_data[0]
                    
                    if embeddings.numel() == 0:
                        species = "Unknown"
                        accuracy = 0.0
                        print("      Species: Unknown (database empty)")
                    else:
                        classification_results = self.classifier.batch_inference([cropped_bgr])
                        
                        if classification_results and len(classification_results) > 0 and classification_results[0]:
                            top_result = classification_results[0][0]
                            species = top_result['name']
                            accuracy = top_result['accuracy']
                            print(f"      Species: {species}")
                            print(f"      Classification confidence: {accuracy:.3f}")
                        else:
                            species = "Unknown"
                            accuracy = 0.0
                            print("      Species: Unknown (no classification result)")
                        
                except Exception as e:
                    species = "Unknown"
                    accuracy = 0.0
                    print(f"      Classification error: {e}")
                    print("      Species: Unknown")
                
                # 3. SEGMENTATION
                print("   🎨 Step 3: Segmentation...")
                try:
                    segmentation_results = self.segmentator.predict(cropped_bgr)
                    if segmentation_results and len(segmentation_results) > 0:
                        mask = segmentation_results[0].mask_polygon(cropped_bgr)
                        print("      Segmentation: Success")
                    else:
                        mask = cropped_rgb  # Fallback
                        print("      Segmentation: Failed, using cropped image")
                        
                except Exception as e:
                    mask = cropped_rgb
                    print(f"      Segmentation error: {e}")
                
                # Simpan hasil
                result = {
                    'fish_id': i + 1,
                    'species': species,
                    'classification_confidence': accuracy,
                    'detection_confidence': box.score,
                    'bounding_box': [int(box.x1), int(box.y1), int(box.x2), int(box.y2)],
                    'cropped_image': cropped_rgb,
                    'segmentation_mask': mask
                }
                results.append(result)
            
            # Visualisasi
            if show_visualization and results:
                self._show_results(img_rgb, results)
            
            return results
            
        except Exception as e:
            print(f"❌ Error in prediction: {e}")
            return []
    
    def _show_results(self, original_img, results):
        """Menampilkan hasil prediksi"""
        try:
            num_fish = len(results)
            fig, axes = plt.subplots(2, num_fish + 1, figsize=(5 * (num_fish + 1), 10))
            
            if num_fish == 1:
                axes = axes.reshape(2, -1)
            elif num_fish == 0:
                return
            
            # Gambar asli dengan bounding boxes
            img_with_boxes = original_img.copy()
            for result in results:
                bbox = result['bounding_box']
                x1, y1, x2, y2 = bbox
                # Gunakan OpenCV untuk menggambar (format BGR)
                img_bgr = cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR)
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(img_bgr, f"{result['species']}", 
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                img_with_boxes = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            axes[0, 0].imshow(img_with_boxes)
            axes[0, 0].set_title("Original + Detections", fontsize=12)
            axes[0, 0].axis('off')
            
            axes[1, 0].imshow(original_img)
            axes[1, 0].set_title("Original Image", fontsize=12)
            axes[1, 0].axis('off')
            
            # Hasil untuk setiap ikan
            for i, result in enumerate(results):
                col = i + 1
                
                # Cropped image
                axes[0, col].imshow(result['cropped_image'])
                title1 = f"Fish #{result['fish_id']}\n{result['species']}\nAcc: {result['classification_confidence']:.3f}"
                axes[0, col].set_title(title1, fontsize=10)
                axes[0, col].axis('off')
                
                # Segmentation mask
                axes[1, col].imshow(result['segmentation_mask'])
                title2 = f"Segmentation\nDet Conf: {result['detection_confidence']:.3f}"
                axes[1, col].set_title(title2, fontsize=10)
                axes[1, col].axis('off')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"⚠️ Cannot show visualization: {e}")

def demo_training():
    """Demo proses training (menambah embedding)"""
    print("\n" + "="*60)
    print("🎓 DEMO TRAINING - Adding New Fish Species")
    print("="*60)
    
    system = SimpleFishSystem()
    system.show_database_info()
    
    # Contoh menambah spesies baru
    test_images = [
        ("images/bandeng.jpg", "Bandeng"),
        ("images/bandeng1.jpg", "Bandeng"),
        ("images/bandeng2.jpg", "Bandeng"),
        ("images/bandeng3.jpg", "Bandeng"),
        ("images/bandeng4.jpg", "Bandeng")
    ]
    
    for img_path, species in test_images:
        if os.path.exists(img_path):
            print(f"\n📚 Training example: {species}")
            success = system.add_new_fish_embedding(img_path, species)
            if success:
                print(f"✅ Successfully processed {species}")
            else:
                print(f"❌ Failed to process {species}")
        else:
            print(f"⚠️ Image not found: {img_path}")

def demo_prediction():
    """Demo proses prediction"""
    print("\n" + "="*60)
    print("🔮 DEMO PREDICTION - Fish Recognition")
    print("="*60)
    
    system = SimpleFishSystem()
    
    # Test dengan berbagai gambar
    test_images = [
        "images/bandeng5.jpg",
        "images/bandeng6.jpg",
        "images/bandeng7.jpg"
    ]
    
    for test_img in test_images:
        if os.path.exists(test_img):
            print(f"\n🧪 Testing: {test_img}")
            results = system.predict_fish_complete(test_img, show_visualization=True)
            
            if results:
                print(f"\n📋 Summary for {os.path.basename(test_img)}:")
                for result in results:
                    print(f"   Fish #{result['fish_id']}: {result['species']} "
                          f"(cls_conf: {result['classification_confidence']:.3f}, "
                          f"det_conf: {result['detection_confidence']:.3f})")
            else:
                print("   No fish detected or prediction failed")
        else:
            print(f"⚠️ Test image not found: {test_img}")

def main():
    """Main demo function"""
    print("🐟 Fish Recognition System - Complete Demo")
    print("="*60)
    
    # Demo 1: Training (menambah embedding)
    demo_training()
    
    # Demo 2: Prediction (test recognition)
    demo_prediction()
    
    print("\n" + "="*60)
    print("🎉 Demo completed!")
    print("="*60)
    print("\n📝 Summary:")
    print("   1. Training: Menunjukkan cara menambah embedding baru")
    print("   2. Prediction: Detection + Classification + Segmentation")
    print("   3. Untuk implementasi production, perlu modifikasi database structure")

if __name__ == "__main__":
    main()