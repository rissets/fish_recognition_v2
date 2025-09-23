#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contoh Sederhana Training dan Prediction
"""

import os
import cv2
import torch
from models.classification.inference import EmbeddingClassifier
from models.detection.inference import YOLOInference
from models.segmentation.inference import Inference

def contoh_training():
    """Contoh menambah spesies baru ke database"""
    print("=== CONTOH TRAINING ===")
    
    # Load model
    classifier = EmbeddingClassifier('models/classification/model.ts', 'models/classification/database.pt')
    detector = YOLOInference('models/detection/model.ts', imsz=(640, 640), conf_threshold=0.9, nms_threshold=0.3, yolo_ver='v10')
    
    # Gambar yang akan ditambahkan
    image_path = 'images/bandeng.jpg'
    species_name = 'mujair'
    
    if not os.path.exists(image_path):
        print(f"Gambar {image_path} tidak ditemukan!")
        return
    
    print(f"Menambah spesies '{species_name}' dari gambar: {image_path}")
    
    # Load gambar
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Detection - cari ikan dalam gambar
    boxes = detector.predict(img_rgb)[0]
    if not boxes:
        print("Tidak ada ikan terdeteksi!")
        return
    
    print(f"Ikan terdeteksi dengan confidence: {boxes[0].score:.3f}")
    
    # Crop gambar berdasarkan detection
    cropped_bgr = boxes[0].get_mask_BGR()
    
    # Extract embedding
    embedding = classifier.get_embedding(cropped_bgr)
    
    # Load database existing
    db = torch.load('models/classification/database.pt')
    print(f"Database saat ini memiliki {len(db['labels'])} embedding")
    
    # Tambah embedding baru
    db['embeddings'].append(embedding)
    db['labels'].append(species_name)
    
    # Save database
    torch.save(db, 'models/classification/database.pt')
    
    print(f"✓ Berhasil menambah '{species_name}' ke database!")
    print(f"✓ Database sekarang memiliki {len(db['labels'])} embedding")

def contoh_prediction():
    """Contoh prediksi gambar"""
    print("\n=== CONTOH PREDICTION ===")
    
    # Load model (reload untuk database yang baru)
    classifier = EmbeddingClassifier('models/classification/model.ts', 'models/classification/database.pt')
    detector = YOLOInference('models/detection/model.ts', imsz=(640, 640), conf_threshold=0.9, nms_threshold=0.3, yolo_ver='v10')
    segmentator = Inference(model_path='models/segmentation/model.ts', image_size=416)
    
    # Gambar yang akan diprediksi
    image_path = 'images/mujair2.jpg'
    
    if not os.path.exists(image_path):
        print(f"Gambar {image_path} tidak ditemukan!")
        return
    
    print(f"Memprediksi gambar: {image_path}")
    
    # Load gambar
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # 1. DETECTION
    print("1. Detection...")
    boxes = detector.predict(img_rgb)[0]
    if not boxes:
        print("Tidak ada ikan terdeteksi!")
        return
    
    print(f"   Detected {len(boxes)} fish(es)")
    
    for i, box in enumerate(boxes):
        print(f"\n   Fish #{i+1}:")
        print(f"   - Detection confidence: {box.score:.3f}")
        
        # Crop gambar
        cropped_bgr = box.get_mask_BGR()
        
        # 2. CLASSIFICATION
        print("   2. Classification...")
        classification_results = classifier.batch_inference([cropped_bgr])
        
        if classification_results and classification_results[0]:
            top_result = classification_results[0][0]
            species_name = top_result['name']
            accuracy = top_result['accuracy']
            print(f"   - Species: {species_name}")
            print(f"   - Classification accuracy: {accuracy:.3f}")
        else:
            print("   - Species: Unknown")
        
        # 3. SEGMENTATION
        print("   3. Segmentation...")
        try:
            segmentation_results = segmentator.predict(cropped_bgr)
            if segmentation_results:
                print("   - Segmentation: Success")
                # segmentation_result = segmentation_results[0]
                # mask = segmentation_result.mask_polygon(cropped_bgr)
            else:
                print("   - Segmentation: Failed")
        except Exception as e:
            print(f"   - Segmentation error: {e}")

def main():
    """Main function untuk demo"""
    print("DEMO TRAINING DAN PREDICTION FISH RECOGNITION")
    print("=" * 50)
    
    # 1. Training - tambah spesies baru
    contoh_training()
    
    # 2. Prediction - test model
    contoh_prediction()
    
    print("\n" + "=" * 50)
    print("DEMO SELESAI!")

if __name__ == "__main__":
    main()