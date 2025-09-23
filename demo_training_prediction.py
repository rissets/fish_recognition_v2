#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo Training dan Prediction untuk Fish Recognition
Menggunakan embedding approach tanpa training ulang model
"""

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
from models.classification.inference import EmbeddingClassifier
from models.detection.inference import YOLOInference
from models.segmentation.inference import Inference
import json

class FishEmbeddingSystem:
    def __init__(self):
        # Path model
        self.classifier_model_path = 'models/classification/model.ts'
        self.classifier_db_path = 'models/classification/database.pt'
        self.detector_model_path = 'models/detection/model.ts'
        self.segmentation_model_path = 'models/segmentation/model.ts'
        
        # Inisialisasi model
        print("Loading models...")
        
        # Pastikan database valid sebelum load classifier
        self._ensure_valid_database()
        
        self.classifier = EmbeddingClassifier(
            self.classifier_model_path, 
            self.classifier_db_path
        )
        
        self.detector = YOLOInference(
            self.detector_model_path,
            imsz=(640, 640),
            conf_threshold=0.7,  # Turunkan threshold untuk deteksi lebih sensitif
            nms_threshold=0.3,
            yolo_ver='v10'
        )
        
        self.segmentator = Inference(
            model_path=self.segmentation_model_path,
            image_size=416
        )
        print("Models loaded successfully!")
    
    def _ensure_valid_database(self):
        """Pastikan database dalam kondisi valid dan tidak rusak"""
        try:
            if os.path.exists(self.classifier_db_path):
                # Coba load database
                db = torch.load(self.classifier_db_path)
                
                # Validasi struktur database
                required_keys = ['embeddings', 'labels', 'image_ids']
                for key in required_keys:
                    if key not in db:
                        db[key] = []
                
                # Pastikan semua list memiliki panjang yang sama
                max_len = max(len(db['embeddings']), len(db['labels']), len(db['image_ids']))
                
                # Potong ke panjang minimum untuk konsistensi
                min_len = min(len(db['embeddings']), len(db['labels']), len(db['image_ids']))
                if min_len != max_len:
                    print(f"⚠️ Database inconsistent, fixing... ({max_len} -> {min_len})")
                    db['embeddings'] = db['embeddings'][:min_len]
                    db['labels'] = db['labels'][:min_len]
                    db['image_ids'] = db['image_ids'][:min_len]
                
                # Save database yang sudah diperbaiki
                torch.save(db, self.classifier_db_path)
                print(f"✅ Database validated with {min_len} entries")
                
            else:
                # Buat database kosong jika tidak ada
                print("📦 Creating new embedding database...")
                empty_db = {
                    'embeddings': [],
                    'labels': [],
                    'image_ids': []
                }
                torch.save(empty_db, self.classifier_db_path)
                print("✅ New database created")
                
        except Exception as e:
            print(f"❌ Database corrupted, creating new one: {e}")
            # Backup database lama jika ada
            if os.path.exists(self.classifier_db_path):
                backup_path = self.classifier_db_path + '.backup'
                os.rename(self.classifier_db_path, backup_path)
                print(f"📋 Corrupted database backed up to {backup_path}")
            
            # Buat database baru
            empty_db = {
                'embeddings': [],
                'labels': [],
                'image_ids': []
            }
            torch.save(empty_db, self.classifier_db_path)
            print("✅ New database created")
    
    def add_fish_species_from_folder(self, folder_path, species_name, max_images=5):
        """
        Menambahkan spesies ikan baru dari folder gambar
        
        Args:
            folder_path: Path ke folder berisi gambar ikan
            species_name: Nama spesies ikan
            max_images: Maksimal jumlah gambar yang akan diproses
        """
        print(f"\n=== TRAINING: Menambah spesies '{species_name}' ===")
        
        # Cari semua file gambar di folder
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(folder_path, ext)))
        
        if not image_files:
            print(f"Tidak ada gambar ditemukan di folder: {folder_path}")
            return False
        
        print(f"Ditemukan {len(image_files)} gambar, akan memproses maksimal {max_images}")
        
        # Load database existing
        db = torch.load(self.classifier_db_path)
        initial_count = len(db['labels'])
        
        embeddings_added = 0
        for i, image_path in enumerate(image_files[:max_images]):
            print(f"Processing {i+1}/{min(len(image_files), max_images)}: {os.path.basename(image_path)}")
            
            try:
                # Load dan preprocess gambar
                img_bgr = cv2.imread(image_path)
                if img_bgr is None:
                    print(f"  Error: Tidak bisa load gambar {image_path}")
                    continue
                    
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                
                # Detection - cari ikan dalam gambar
                boxes = self.detector.predict(img_rgb)[0]
                if not boxes:
                    print(f"  Warning: Tidak ada ikan terdeteksi di {image_path}")
                    continue
                
                # Ambil deteksi dengan confidence tertinggi
                best_box = max(boxes, key=lambda x: x.score)
                print(f"  Fish detected with confidence: {best_box.score:.3f}")
                
                # Crop gambar berdasarkan bounding box
                cropped_bgr = best_box.get_mask_BGR()
                
                # Extract embedding menggunakan classifier
                embedding = self.classifier.get_embedding(cropped_bgr)
                
                # Tambahkan ke database
                db['embeddings'].append(embedding)
                db['labels'].append(species_name)
                
                # Tambahkan image_id jika tidak ada
                if 'image_ids' not in db:
                    db['image_ids'] = []
                
                # Generate unique image ID
                import uuid
                image_id = str(uuid.uuid4())
                db['image_ids'].append(image_id)
                
                embeddings_added += 1
                
                print(f"  ✓ Embedding berhasil ditambahkan")
                
            except Exception as e:
                print(f"  Error processing {image_path}: {str(e)}")
                continue
        
        # Save database yang sudah diupdate
        if embeddings_added > 0:
            torch.save(db, self.classifier_db_path)
            print(f"\n✓ Training selesai!")
            print(f"✓ {embeddings_added} embedding baru ditambahkan untuk spesies '{species_name}'")
            print(f"✓ Total embedding di database: {initial_count} → {len(db['labels'])}")
            
            # Reload classifier dengan database yang baru
            self.classifier = EmbeddingClassifier(
                self.classifier_model_path, 
                self.classifier_db_path
            )
            return True
        else:
            print(f"✗ Tidak ada embedding yang berhasil ditambahkan")
            return False
    
    def predict_fish_comprehensive(self, image_path, visualize=True):
        """
        Prediksi komprehensif: detection + classification + segmentation
        
        Args:
            image_path: Path ke gambar yang akan diprediksi
            visualize: Apakah akan menampilkan visualisasi hasil
        """
        print(f"\n=== PREDICTION: {os.path.basename(image_path)} ===")
        
        # Load gambar
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            print(f"Error: Tidak bisa load gambar {image_path}")
            return None
            
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        original_img = img_rgb.copy()
        
        # 1. DETECTION
        print("1. Fish Detection...")
        boxes = self.detector.predict(img_rgb)[0]
        if not boxes:
            print("   Tidak ada ikan terdeteksi!")
            return None
        
        print(f"   Detected {len(boxes)} fish(es)")
        
        results = []
        
        for i, box in enumerate(boxes):
            print(f"\n   Fish #{i+1}:")
            print(f"   - Confidence: {box.score:.3f}")
            print(f"   - Bounding box: {[box.x1, box.y1, box.x2, box.y2]}")
            
            # Crop gambar berdasarkan detection
            cropped_bgr = box.get_mask_BGR()
            cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
            
            # 2. CLASSIFICATION
            print("   2. Classification...")
            try:
                classification_results = self.classifier.batch_inference([cropped_bgr])
                
                if classification_results and len(classification_results) > 0 and classification_results[0]:
                    top_result = classification_results[0][0]
                    species_name = top_result['name']
                    accuracy = top_result['accuracy']
                    print(f"   - Species: {species_name}")
                    print(f"   - Accuracy: {accuracy:.3f}")
                else:
                    species_name = "Unknown"
                    accuracy = 0.0
                    print("   - Species: Unknown (no classification result)")
            except Exception as e:
                species_name = "Unknown"
                accuracy = 0.0
                print(f"   - Classification error: {e}")
                print("   - Species: Unknown")
            
            # 3. SEGMENTATION
            print("   3. Segmentation...")
            try:
                segmentation_results = self.segmentator.predict(cropped_bgr)
                if segmentation_results:
                    segmentation_result = segmentation_results[0]
                    mask = segmentation_result.mask_polygon(cropped_bgr)
                    print("   - Segmentation: Success")
                else:
                    mask = cropped_rgb  # Fallback ke cropped image
                    print("   - Segmentation: Failed, using cropped image")
            except Exception as e:
                mask = cropped_rgb
                print(f"   - Segmentation error: {e}")
            
            # Simpan hasil
            result = {
                'fish_id': i + 1,
                'species': species_name,
                'accuracy': accuracy,
                'confidence': box.score,
                'bbox': [box.x1, box.y1, box.x2, box.y2],
                'cropped_image': cropped_rgb,
                'segmented_mask': mask
            }
            results.append(result)
        
        # VISUALISASI
        if visualize:
            self._visualize_results(original_img, results)
        
        return results
    
    def _visualize_results(self, original_img, results):
        """Visualisasi hasil detection, classification, dan segmentation"""
        
        num_fish = len(results)
        fig, axes = plt.subplots(2, num_fish + 1, figsize=(4 * (num_fish + 1), 8))
        
        if num_fish == 1:
            axes = axes.reshape(2, -1)
        
        # Gambar original dengan bounding boxes
        img_with_boxes = original_img.copy()
        for result in results:
            bbox = result['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_with_boxes, f"{result['species']}", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        axes[0, 0].imshow(img_with_boxes)
        axes[0, 0].set_title("Original + Detection")
        axes[0, 0].axis('off')
        
        axes[1, 0].imshow(original_img)
        axes[1, 0].set_title("Original Image")
        axes[1, 0].axis('off')
        
        # Hasil untuk setiap ikan
        for i, result in enumerate(results):
            col = i + 1
            
            # Cropped image
            axes[0, col].imshow(result['cropped_image'])
            axes[0, col].set_title(f"Fish #{result['fish_id']}\n{result['species']}\nAcc: {result['accuracy']:.3f}")
            axes[0, col].axis('off')
            
            # Segmented mask
            axes[1, col].imshow(result['segmented_mask'])
            axes[1, col].set_title(f"Segmentation\nConf: {result['confidence']:.3f}")
            axes[1, col].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def batch_predict_folder(self, folder_path, max_images=5):
        """Prediksi batch untuk semua gambar dalam folder"""
        print(f"\n=== BATCH PREDICTION: {folder_path} ===")
        
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(folder_path, ext)))
        
        if not image_files:
            print(f"Tidak ada gambar ditemukan di folder: {folder_path}")
            return
        
        print(f"Akan memproses {min(len(image_files), max_images)} gambar...")
        
        for i, image_path in enumerate(image_files[:max_images]):
            print(f"\n--- Image {i+1}/{min(len(image_files), max_images)} ---")
            self.predict_fish_comprehensive(image_path, visualize=False)
    
    def show_database_stats(self):
        """Menampilkan statistik database embedding"""
        print("\n=== DATABASE STATISTICS ===")
        
        db = torch.load(self.classifier_db_path)
        labels = db['labels']
        embeddings = db['embeddings']
        
        print(f"Total embeddings: {len(embeddings)}")
        print(f"Total labels: {len(labels)}")
        
        # Hitung jumlah per spesies
        from collections import Counter
        species_count = Counter(labels)
        
        print("\nJumlah embedding per spesies:")
        for species, count in sorted(species_count.items()):
            print(f"  {species}: {count}")
        
        print(f"\nTotal unique species: {len(species_count)}")


def main():
    """Contoh penggunaan sistem"""
    
    # Inisialisasi sistem
    fish_system = FishEmbeddingSystem()
    
    # Tampilkan statistik database awal
    fish_system.show_database_stats()
    
    print("\n" + "="*60)
    print("CONTOH PENGGUNAAN FISH EMBEDDING SYSTEM")
    print("="*60)
    
    # 1. CONTOH TRAINING - Menambah spesies baru
    print("\n1. TRAINING - Menambah spesies baru ke database")
    print("-" * 50)
    
    # Contoh: tambahkan spesies "mujair" dari gambar yang ada
    mujair_images = ['images/mujair1.jpg', 'images/mujair2.jpg', 'images/mujair3.jpg']
    
    # Cek apakah file gambar ada
    existing_images = [img for img in mujair_images if os.path.exists(img)]
    
    if existing_images:
        print(f"Menambah spesies 'mujair' menggunakan {len(existing_images)} gambar...")
        
        # Buat folder temporary untuk demo
        temp_folder = 'temp_mujair'
        os.makedirs(temp_folder, exist_ok=True)
        
        # Copy gambar ke folder temporary
        import shutil
        for img in existing_images:
            shutil.copy(img, temp_folder)
        
        # Tambahkan ke database
        fish_system.add_fish_species_from_folder(temp_folder, 'mujair', max_images=3)
        
        # Cleanup
        shutil.rmtree(temp_folder)
    else:
        print("Gambar mujair tidak ditemukan, skip training demo")
    
    # 2. CONTOH PREDICTION
    print("\n\n2. PREDICTION - Test gambar")
    print("-" * 50)
    
    # Test dengan gambar yang ada
    test_images = ['images/mujair1.jpg', 'image_testing/mujair4.jpg']
    
    for test_img in test_images:
        if os.path.exists(test_img):
            results = fish_system.predict_fish_comprehensive(test_img, visualize=True)
            
            if results:
                print(f"\nHasil prediksi untuk {test_img}:")
                for result in results:
                    print(f"  Fish #{result['fish_id']}: {result['species']} "
                          f"(accuracy: {result['accuracy']:.3f}, "
                          f"confidence: {result['confidence']:.3f})")
        else:
            print(f"File {test_img} tidak ditemukan")
    
    # 3. BATCH PREDICTION
    print("\n\n3. BATCH PREDICTION - Test folder")
    print("-" * 50)
    
    if os.path.exists('images'):
        fish_system.batch_predict_folder('images', max_images=3)
    
    # Tampilkan statistik database akhir
    print("\n" + "="*60)
    fish_system.show_database_stats()
    
    print("\n" + "="*60)
    print("DEMO SELESAI!")
    print("="*60)


if __name__ == "__main__":
    main()