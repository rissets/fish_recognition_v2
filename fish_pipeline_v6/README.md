# Fish Pipeline V6

## Fitur
- Preprocessing (1 gambar → 10 gambar, segmentasi, augmentasi)
- Training CNN sederhana
- Embedding + Classification
- Detection (YOLO) & Segmentation (FPN)
- Menggunakan model yang sudah ada di folder `models/classification`, `models/detection`, `models/segmentation`

## Struktur Folder
- `preprocessing.py` : Preprocessing & augmentasi
- `train_cnn.py` : Training CNN sederhana
- `embedding_classification.py` : Embedding + klasifikasi
- `detection_segmentation.py` : Deteksi & segmentasi ikan
- `requirements.txt` : Dependensi

## Cara Pakai
1. Install dependencies: `pip install -r requirements.txt`
2. Jalankan preprocessing: `python preprocessing.py`
3. Training: `python train_cnn.py`
4. Embedding & klasifikasi: `python embedding_classification.py`
5. Deteksi & segmentasi: `python detection_segmentation.py`
