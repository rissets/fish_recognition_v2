# Script untuk menambahkan dataset baru dan update embedding database
# Pastikan dependencies sudah terinstall: cv2, torch, PIL, numpy, dll

import os
import json
import torch
import cv2
import numpy as np
from PIL import Image
from models.classification.inference import EmbeddingClassifier
from models.detection.inference import YOLOInference
from models.segmentation.inference import Inference

# Path model dan database
CLASSIFIER_MODEL_PATH = 'models/classification/model.ts'
CLASSIFIER_DB_PATH = 'models/classification/database.pt'
DETECTOR_MODEL_PATH = 'models/detection/model.ts'
SEGMENTATION_MODEL_PATH = 'models/segmentation/model.ts'

# Inisialisasi model
classifier = EmbeddingClassifier(CLASSIFIER_MODEL_PATH, CLASSIFIER_DB_PATH)
detector = YOLOInference(DETECTOR_MODEL_PATH, imsz=(640, 640), conf_threshold=0.9, nms_threshold=0.3, yolo_ver='v10')
segmentator = Inference(model_path=SEGMENTATION_MODEL_PATH, image_size=416)

# Fungsi untuk menambah embedding ke database
# database.pt format: {'embeddings': [np.array], 'labels': [str]}
def add_new_fish_to_db(image_path, species_name):
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # Detection
    boxes = detector.predict(img_rgb)[0]
    if not boxes:
        print('No fish detected!')
        return False
    # Ambil box pertama
    box = boxes[0]
    cropped_bgr = box.get_mask_BGR()
    # Segmentation
    segmented = segmentator.predict(cropped_bgr)[0]
    mask = segmented.mask_polygon(cropped_bgr)
    # Embedding
    embedding = classifier.get_embedding(cropped_bgr)
    # Load database
    db = torch.load(CLASSIFIER_DB_PATH)
    db['embeddings'].append(embedding)
    db['labels'].append(species_name)
    torch.save(db, CLASSIFIER_DB_PATH)
    print(f"Added {species_name} to database!")
    return True

# Fungsi prediksi
# Output: label, akurasi, segmentasi

def predict_fish(image_path):
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    boxes = detector.predict(img_rgb)[0]
    results = []
    for box in boxes:
        cropped_bgr = box.get_mask_BGR()
        segmented = segmentator.predict(cropped_bgr)[0]
        mask = segmented.mask_polygon(cropped_bgr)
        classification = classifier.batch_inference([cropped_bgr])[0]
        label = classification[0]['name'] if classification else 'Unknown'
        acc = classification[0]['accuracy'] if classification else 0.0
        results.append({'label': label, 'accuracy': acc, 'mask': mask})
    return results

if __name__ == "__main__":
    # Contoh penggunaan
    # Tambah ikan baru
    # add_new_fish_to_db('images/mujair1.jpg', 'mujair')
    # Prediksi
    results = predict_fish('images/mujair1.jpg')
    for r in results:
        print(f"Label: {r['label']}, Akurasi: {r['accuracy']}")
        # Visualisasi segmentasi
        Image.fromarray(r['mask']).show()
