#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contoh Training dan Prediction - Version yang dapat berjalan
"""

import os
import cv2
import torch
from models.classification.inference import EmbeddingClassifier
from models.detection.inference import YOLOInference
from models.segmentation.inference import Inference

def training_example():
    """Contoh training dengan threshold detection yang lebih rendah"""
    print("=== TRAINING EXAMPLE ===")
    
    # Load model dengan threshold yang lebih rendah untuk training
    detector = YOLOInference('models/detection/model.ts', imsz=(640, 640), conf_threshold=0.5, nms_threshold=0.3, yolo_ver='v10')
    classifier = EmbeddingClassifier('models/classification/model.ts', 'models/classification/database.pt')
    
    # Gambar training
    image_path = 'images/bandeng.jpg'
    species_name = 'Bandeng'
    
    if not os.path.exists(image_path):
        print(f"Gambar {image_path} tidak ditemukan!")
        return False
    
    print(f"Training: Menambah spesies '{species_name}' dari: {image_path}")
    
    # Load dan preprocess gambar
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Detection dengan threshold rendah
    boxes = detector.predict(img_rgb)[0]
    
    if not boxes:
        print("❌ Tidak ada ikan terdeteksi dengan threshold 0.5")
        return False
    
    print(f"✅ Detected {len(boxes)} fish(es)")
    
    # Ambil fish dengan confidence tertinggi
    best_box = max(boxes, key=lambda x: x.score)
    print(f"✅ Menggunakan fish dengan confidence: {best_box.score:.3f}")
    
    # Crop gambar
    cropped_bgr = best_box.get_mask_BGR()
    
    # Extract embedding
    # Gunakan model untuk extract embedding
    from PIL import Image
    import torchvision.transforms as transforms
    
    # Preprocess image untuk model
    loader = transforms.Compose([
        transforms.Resize((224, 224), Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Convert BGR ke RGB untuk PIL
    cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(cropped_rgb)
    input_tensor = loader(pil_image).unsqueeze(0)
    
    # Extract embedding menggunakan model
    with torch.no_grad():
        embedding, _ = classifier.model(input_tensor)
        embedding = embedding.squeeze(0)  # Remove batch dimension
    
    print(f"✅ Embedding extracted, shape: {embedding.shape}")
    
    # Load database dan backup
    db_path = 'models/classification/database.pt'
    backup_path = 'models/classification/database_backup.pt'
    
    # Backup database
    import shutil
    shutil.copy(db_path, backup_path)
    print(f"✅ Database backup created: {backup_path}")
    
    # Load database
    db = torch.load(db_path)
    
    if isinstance(db, list) and len(db) >= 2:
        embeddings_tensor = db[0]  # Tensor embeddings
        labels_list = db[1]        # List of integer IDs
        original_count = len(labels_list)
        
        # Untuk demo, kita akan menggunakan ID baru (maksimal ID + 1)
        if labels_list:
            new_id = max(labels_list) + 1
        else:
            new_id = 0
        
        print(f"✅ Menggunakan ID {new_id} untuk spesies '{species_name}'")
        
        # Tambah embedding baru
        # Concatenate tensor embedding ke embeddings_tensor yang ada
        embeddings_tensor = torch.cat([embeddings_tensor, embedding.unsqueeze(0)], dim=0)
        labels_list.append(new_id)
        
        # Update database
        db[0] = embeddings_tensor
        db[1] = labels_list
        
        # Save database
        torch.save(db, db_path)
        
        print(f"✅ Training berhasil!")
        print(f"✅ Database: {original_count} → {len(labels_list)} embeddings")
        print(f"✅ Spesies '{species_name}' berhasil ditambahkan dengan ID {new_id}")
    else:
        print("❌ Format database tidak dikenal!")
        return False
    
    return True

def prediction_example():
    """Contoh prediction dengan model yang sudah diupdate"""
    print("\n=== PREDICTION EXAMPLE ===")
    
    # Load model untuk prediction (threshold normal)
    detector = YOLOInference('models/detection/model.ts', imsz=(640, 640), conf_threshold=0.9, nms_threshold=0.3, yolo_ver='v10')
    classifier = EmbeddingClassifier('models/classification/model.ts', 'models/classification/database.pt')
    segmentator = Inference(model_path='models/segmentation/model.ts', image_size=416)
    
    # Test gambar
    test_images = ['images/bandeng.jpg', 'images/bandeng.jpg']
    
    for image_path in test_images:
        if not os.path.exists(image_path):
            print(f"❌ Gambar {image_path} tidak ditemukan!")
            continue
            
        print(f"\n🔍 Testing: {image_path}")
        
        # Load gambar
        img_bgr = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Detection
        boxes = detector.predict(img_rgb)[0]
        
        if not boxes:
            print("❌ Tidak ada ikan terdeteksi")
            continue
        
        print(f"✅ Detected {len(boxes)} fish(es)")
        
        # Proses setiap fish yang terdeteksi
        for i, box in enumerate(boxes[:3]):  # Max 3 fish
            print(f"\n   🐟 Fish #{i+1}:")
            print(f"      Detection confidence: {box.score:.3f}")
            
            # Crop
            cropped_bgr = box.get_mask_BGR()
            cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
            
            # Classification
            classification_results = classifier.batch_inference([cropped_bgr])
            
            if classification_results and classification_results[0]:
                # Ambil top 3 predictions
                top_results = classification_results[0][:3]
                print(f"      🏷️  Top predictions:")
                for j, result in enumerate(top_results):
                    species = result['name']
                    accuracy = result['accuracy']
                    print(f"         {j+1}. {species} (accuracy: {accuracy:.3f})")
            else:
                print("      🏷️  Species: Unknown")
            
            # Segmentation
            try:
                segmentation_results = segmentator.predict(cropped_bgr)
                if segmentation_results:
                    print("      ✂️  Segmentation: Success")
                else:
                    print("      ✂️  Segmentation: Failed")
            except Exception as e:
                print(f"      ✂️  Segmentation error: {e}")

def show_database_info():
    """Tampilkan informasi database"""
    print("\n=== DATABASE INFO ===")
    
    db_path = 'models/classification/database.pt'
    if not os.path.exists(db_path):
        print("❌ Database tidak ditemukan!")
        return
    
    db = torch.load(db_path)
    
    if isinstance(db, list) and len(db) >= 2:
        embeddings = db[0]  # Tensor embeddings
        labels = db[1]      # List of integer IDs
        
        print(f"📊 Total embeddings: {len(embeddings)}")
        print(f"📊 Total labels: {len(labels)}")
        
        # Count per species ID
        from collections import Counter
        label_count = Counter(labels)
        
        print(f"📊 Unique species IDs: {len(label_count)}")
        print("\n📋 Species ID distribution (top 10):")
        
        # Show top 10 most common species IDs
        for label_id, count in label_count.most_common(10):
            print(f"   ID {label_id}: {count} embeddings")
        
        if len(label_count) > 10:
            print(f"   ... and {len(label_count) - 10} more species")
    else:
        print("❌ Database format tidak dikenal!")
        print(f"Database type: {type(db)}")
        if isinstance(db, list):
            print(f"Database length: {len(db)}")

def main():
    """Main demo function"""
    print("🐟 FISH RECOGNITION TRAINING & PREDICTION DEMO")
    print("=" * 60)
    
    # 1. Show initial database info
    show_database_info()
    
    # 2. Training example
    success = training_example()
    
    if success:
        # 3. Show updated database info
        show_database_info()
        
        # 4. Prediction example
        prediction_example()
    else:
        print("\n❌ Training gagal, skip prediction demo")
    
    print("\n" + "=" * 60)
    print("🎉 DEMO SELESAI!")
    
    # Info backup
    backup_path = 'models/classification/database_backup.pt'
    if os.path.exists(backup_path):
        print(f"\n💾 Database backup tersedia di: {backup_path}")
        print("💡 Untuk restore: cp database_backup.pt database.pt")

if __name__ == "__main__":
    main()