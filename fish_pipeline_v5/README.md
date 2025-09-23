# Fish Pipeline V5

Sistem fish recognition terintegrasi:
- Preprocessing (1 gambar → 10 gambar)
- Training CNN
- Embedding Classification
- Detection (YOLO)
- Segmentation (FPN)

Menggunakan model yang sudah ada di folder `models/classification`, `models/detection`, dan `models/segmentation`.

## Struktur Folder
- `preprocessing.py` : Preprocessing & augmentasi
- `train_cnn.py` : Training CNN
- `embedding_classification.py` : Klasifikasi embedding
- `detection.py` : Deteksi ikan
- `segmentation.py` : Segmentasi ikan
- `config.json` : Konfigurasi pipeline

## Cara Pakai
1. Install dependencies: `pip install -r requirements.txt`
2. Jalankan preprocessing: `python preprocessing.py`
3. Training: `python train_cnn.py`
4. Embedding classification: `python embedding_classification.py`
5. Detection: `python detection.py`
6. Segmentation: `python segmentation.py`
