# Fish Pipeline V3

## Overview
Pipeline untuk preprocessing, training, embedding classification, detection, dan segmentation ikan. Semua model menggunakan file yang sudah ada di folder `models/classification`, `models/detection`, dan `models/segmentation`.

## Struktur
- preprocessing.py: Preprocessing 1 gambar → 10 versi
- train_classifier.py: Training CNN/embedding classifier
- embedding_inference.py: Embedding-based classification
- detection_inference.py: Deteksi ikan (YOLO)
- segmentation_inference.py: Segmentasi ikan (FPN)
- dataset/: Dataset folder (folder=label)
- output/: Output hasil
- models/: Symlink ke models lama

## Cara Pakai
1. Preprocessing: `python preprocessing.py`
2. Training: `python train_classifier.py`
3. Embedding Classification: `python embedding_inference.py`
4. Detection: `python detection_inference.py`
5. Segmentation: `python segmentation_inference.py`

## Model
- Classification: models/classification/model.ts, database.pt
- Detection: models/detection/model.ts
- Segmentation: models/segmentation/model.ts
