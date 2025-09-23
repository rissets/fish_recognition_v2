#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FISH RECOGNITION SYSTEM - FINAL SCRIPT
Sistem untuk training dan prediction ikan menggunakan embedding approach

Fitur:
- Menambah spesies baru tanpa training ulang model
- Detection, Classification, dan Segmentation
- Backup database otomatis
- Format output yang user-friendly

Penggunaan:
1. Training: python fish_system.py train --image path/to/image.jpg --species "nama_spesies"
2. Prediction: python fish_system.py predict --image path/to/image.jpg
3. Info database: python fish_system.py info
"""

import os
import cv2
import torch
import argparse
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from models.classification.inference import EmbeddingClassifier
from models.detection.inference import YOLOInference
from models.segmentation.inference import Inference

class FishRecognitionSystem:
    def __init__(self):
        """Inisialisasi sistem fish recognition"""
        print("🐟 Inisialisasi Fish Recognition System...")
        
        # Path model dan database
        self.classifier_model = 'models/classification/model.ts'
        self.classifier_db = 'models/classification/database.pt'
        self.detector_model = 'models/detection/model.ts'
        self.segmentation_model = 'models/segmentation/model.ts'
        
        # Load semua model
        print("   Loading Classification Model...")
        self.classifier = EmbeddingClassifier(self.classifier_model, self.classifier_db)
        
        print("   Loading Detection Model...")
        self.detector = YOLOInference(
            self.detector_model,
            imsz=(640, 640),
            conf_threshold=0.5,  # Threshold lebih rendah untuk training
            nms_threshold=0.3,
            yolo_ver='v10'
        )
        
        print("   Loading Segmentation Model...")
        self.segmentator = Inference(model_path=self.segmentation_model, image_size=416)
        
        # Setup preprocessing untuk extract embedding
        self.embedding_loader = transforms.Compose([
            transforms.Resize((224, 224), Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        print("✅ Sistem berhasil diinisialisasi!")
    
    def extract_embedding(self, image_bgr):
        """Extract embedding dari gambar ikan"""
        # Convert BGR ke RGB untuk PIL
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Preprocess
        input_tensor = self.embedding_loader(pil_image).unsqueeze(0)
        
        # Extract embedding
        with torch.no_grad():
            embedding, _ = self.classifier.model(input_tensor)
            embedding = embedding.squeeze(0)  # Remove batch dimension
        
        return embedding
    
    def backup_database(self):
        """Backup database sebelum modifikasi"""
        backup_path = 'models/classification/database_backup.pt'
        import shutil
        shutil.copy(self.classifier_db, backup_path)
        print(f"✅ Database backup: {backup_path}")
        return backup_path
    
    def add_species(self, image_path, species_name):
        """Menambah spesies baru ke database"""
        print(f"\n🎯 TRAINING: Menambah spesies '{species_name}'")
        print(f"📷 Gambar: {image_path}")
        
        if not os.path.exists(image_path):
            print(f"❌ Gambar tidak ditemukan: {image_path}")
            return False
        
        # Load dan preprocess gambar
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            print(f"❌ Tidak bisa load gambar: {image_path}")
            return False
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Detection
        print("🔍 Mendeteksi ikan...")
        boxes = self.detector.predict(img_rgb)[0]
        
        if not boxes:
            print("❌ Tidak ada ikan terdeteksi!")
            return False
        
        print(f"✅ Detected {len(boxes)} fish(es)")
        
        # Ambil fish dengan confidence tertinggi
        best_box = max(boxes, key=lambda x: x.score)
        print(f"✅ Menggunakan fish dengan confidence: {best_box.score:.3f}")
        
        # Crop gambar
        cropped_bgr = best_box.get_mask_BGR()
        
        # Extract embedding
        print("🧠 Extracting embedding...")
        embedding = self.extract_embedding(cropped_bgr)
        print(f"✅ Embedding shape: {embedding.shape}")
        
        # Backup database
        backup_path = self.backup_database()
        
        # Load database dan tambah embedding
        db = torch.load(self.classifier_db)
        
        if isinstance(db, list) and len(db) >= 2:
            embeddings_tensor = db[0]
            labels_list = db[1]
            original_count = len(labels_list)
            
            # Generate ID baru
            if labels_list:
                new_id = max(labels_list) + 1
            else:
                new_id = 0
            
            print(f"✅ Menggunakan ID {new_id} untuk spesies '{species_name}'")
            
            # Tambah embedding dan label
            embeddings_tensor = torch.cat([embeddings_tensor, embedding.unsqueeze(0)], dim=0)
            labels_list.append(new_id)
            
            # Update database
            db[0] = embeddings_tensor
            db[1] = labels_list
            
            # Save database
            torch.save(db, self.classifier_db)
            
            # Reload classifier dengan database baru
            self.classifier = EmbeddingClassifier(self.classifier_model, self.classifier_db)
            
            print(f"✅ Training berhasil!")
            print(f"✅ Database: {original_count} → {len(labels_list)} embeddings")
            print(f"✅ Spesies '{species_name}' ID: {new_id}")
            
            return True
        else:
            print("❌ Format database tidak dikenal!")
            return False
    
    def predict_fish(self, image_path, show_details=True):
        """Prediksi ikan dalam gambar"""
        print(f"\n🔮 PREDICTION: {os.path.basename(image_path)}")
        
        if not os.path.exists(image_path):
            print(f"❌ Gambar tidak ditemukan: {image_path}")
            return None
        
        # Load gambar
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            print(f"❌ Tidak bisa load gambar: {image_path}")
            return None
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Detection
        print("🔍 Detecting fish...")
        # Gunakan threshold yang lebih tinggi untuk prediction
        high_conf_detector = YOLOInference(
            self.detector_model,
            imsz=(640, 640),
            conf_threshold=0.9,  # Threshold tinggi untuk prediction
            nms_threshold=0.3,
            yolo_ver='v10'
        )
        
        boxes = high_conf_detector.predict(img_rgb)[0]
        
        if not boxes:
            print("❌ Tidak ada ikan terdeteksi!")
            return None
        
        print(f"✅ Detected {len(boxes)} fish(es)")
        
        results = []
        
        for i, box in enumerate(boxes):
            print(f"\n   🐟 Fish #{i+1}:")
            print(f"      Detection confidence: {box.score:.3f}")
            
            # Crop gambar
            cropped_bgr = box.get_mask_BGR()
            
            # Classification
            print("      🏷️  Classification...")
            classification_results = self.classifier.batch_inference([cropped_bgr])
            
            if classification_results and classification_results[0]:
                top_results = classification_results[0][:3]  # Top 3
                print("         Top predictions:")
                for j, result in enumerate(top_results):
                    species = result['name']
                    accuracy = result['accuracy']
                    print(f"         {j+1}. {species} (acc: {accuracy:.3f})")
                
                # Hasil terbaik
                best_species = top_results[0]['name']
                best_accuracy = top_results[0]['accuracy']
            else:
                best_species = "Unknown"
                best_accuracy = 0.0
                print("         Unknown species")
            
            # Segmentation
            print("      ✂️  Segmentation...")
            try:
                segmentation_results = self.segmentator.predict(cropped_bgr)
                if segmentation_results:
                    print("         Success")
                    segmentation_success = True
                else:
                    print("         Failed")
                    segmentation_success = False
            except Exception as e:
                print(f"         Error: {e}")
                segmentation_success = False
            
            # Simpan hasil
            result = {
                'fish_id': i + 1,
                'species': best_species,
                'accuracy': best_accuracy,
                'detection_confidence': box.score,
                'bbox': [box.x1, box.y1, box.x2, box.y2],
                'segmentation_success': segmentation_success
            }
            results.append(result)
        
        return results
    
    def show_database_info(self):
        """Tampilkan informasi database"""
        print("\n📊 DATABASE INFO")
        print("=" * 50)
        
        if not os.path.exists(self.classifier_db):
            print("❌ Database tidak ditemukan!")
            return
        
        db = torch.load(self.classifier_db)
        
        if isinstance(db, list) and len(db) >= 2:
            embeddings = db[0]
            labels = db[1]
            
            print(f"📊 Total embeddings: {len(embeddings)}")
            print(f"📊 Total labels: {len(labels)}")
            
            # Count per species ID
            from collections import Counter
            label_count = Counter(labels)
            
            print(f"📊 Unique species: {len(label_count)}")
            print("\n🔢 Top 10 species by count:")
            
            for label_id, count in label_count.most_common(10):
                print(f"   ID {label_id}: {count} embeddings")
            
            if len(label_count) > 10:
                print(f"   ... and {len(label_count) - 10} more species")
        else:
            print("❌ Database format tidak dikenal!")

def main():
    """Main function untuk command line interface"""
    parser = argparse.ArgumentParser(description='Fish Recognition System')
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Command: train
    train_parser = subparsers.add_parser('train', help='Add new fish species')
    train_parser.add_argument('--image', required=True, help='Path to fish image')
    train_parser.add_argument('--species', required=True, help='Name of fish species')
    
    # Command: predict
    predict_parser = subparsers.add_parser('predict', help='Predict fish in image')
    predict_parser.add_argument('--image', required=True, help='Path to image')
    
    # Command: info
    info_parser = subparsers.add_parser('info', help='Show database information')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Inisialisasi sistem
    system = FishRecognitionSystem()
    
    if args.command == 'train':
        success = system.add_species(args.image, args.species)
        if success:
            print("\n✅ Training berhasil!")
        else:
            print("\n❌ Training gagal!")
    
    elif args.command == 'predict':
        results = system.predict_fish(args.image)
        if results:
            print(f"\n📋 SUMMARY:")
            print(f"   Total fish detected: {len(results)}")
            for result in results:
                print(f"   Fish #{result['fish_id']}: {result['species']} "
                      f"(acc: {result['accuracy']:.3f}, conf: {result['detection_confidence']:.3f})")
        else:
            print("\n❌ Tidak ada hasil!")
    
    elif args.command == 'info':
        system.show_database_info()

if __name__ == "__main__":
    # Jika dijalankan langsung tanpa argument, jalankan demo
    import sys
    if len(sys.argv) == 1:
        print("🐟 FISH RECOGNITION SYSTEM - DEMO MODE")
        print("=" * 50)
        
        system = FishRecognitionSystem()
        
        # Show database info
        system.show_database_info()
        
        # Test prediction
        test_images = ['images/mujair1.jpg', 'images/mujair2.jpg']
        for img in test_images:
            if os.path.exists(img):
                system.predict_fish(img)
        
        print("\n" + "=" * 50)
        print("💡 Untuk penggunaan:")
        print("   python fish_system.py train --image path/to/image.jpg --species 'nama_spesies'")
        print("   python fish_system.py predict --image path/to/image.jpg")
        print("   python fish_system.py info")
    else:
        main()