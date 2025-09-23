# 🐟 Fish Pipeline V4

## Fitur Utama
- Preprocessing 1→10 gambar (full image, tidak terpotong)
- Training CNN sederhana
- Embedding & Classification (menggunakan model existing)
- Detection & Segmentation (menggunakan model existing)
- Folder label otomatis
- Pipeline end-to-end

## Struktur Folder
```
fish_pipeline_v4/
├── preprocessing.py         # Preprocessing & augmentasi
├── train_cnn.py            # Training CNN sederhana
├── embedding_classify.py   # Embedding & klasifikasi
├── detection_segment.py    # Deteksi & segmentasi
├── pipeline_demo.py        # Demo end-to-end
├── requirements.txt        # Dependencies
├── README.md               # Dokumentasi
└── output/                 # Hasil processing
```

## Cara Pakai
```bash
cd fish_pipeline_v4/
pip install -r requirements.txt
python pipeline_demo.py
```

## Model yang Digunakan
- Detection: models/detection/model.ts
- Segmentation: models/segmentation/model.ts
- Classification: models/classification/model.ts
