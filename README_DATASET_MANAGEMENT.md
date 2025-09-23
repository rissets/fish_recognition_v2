# Fish Recognition Dataset Management

Script ini memungkinkan Anda untuk menambahkan dataset ikan baru ke dalam sistem fish recognition yang sudah ada tanpa perlu melatih ulang model detection dan segmentation. Sistem menggunakan vector embedding untuk classification.

## File-file Utama

1. **`add_new_fish_dataset.py`** - Script utama untuk menambahkan dataset baru
2. **`dataset_utilities.py`** - Utility functions untuk manajemen dataset
3. **`batch_demo.py`** - Script demo dengan interface yang user-friendly
4. **`species_config_example.json`** - Contoh konfigurasi untuk batch processing

## Cara Penggunaan

### 1. Persiapan Model

Pastikan semua model sudah didownload dengan menjalankan:
```bash
python research_fishial.py
```

### 2. Mode Interaktif (Recommended)

```bash
python batch_demo.py --mode interactive
```

Mode ini menyediakan menu interaktif yang mudah digunakan:
- Lihat database saat ini
- Tambah spesies baru secara interaktif
- Jalankan prediksi demo
- Dan lainnya

### 3. Menambah Single Species

```bash
python batch_demo.py --species-path "dataset/ikan_db_v1/images/mujair" --species-name "Mujair"
```

### 4. Batch Processing

Buat file konfigurasi JSON (lihat `species_config_example.json`), lalu:

```bash
python batch_demo.py --mode batch --config species_config_example.json
```

### 5. Prediction Demo

```bash
python batch_demo.py --mode predict --test-dir "images"
```

## Struktur Dataset

Dataset harus diorganisir dalam struktur folder seperti ini:
```
dataset/
  ikan_db_v1/
    images/
      species-name-1/
        image1.jpg
        image2.jpg
        ...
      species-name-2/
        image1.jpg
        image2.jpg
        ...
```

## Fitur Utama

### 1. Vector Embedding Classification
- Menggunakan model classification yang sudah ada untuk extract features
- Membuat database embedding tanpa melatih ulang
- Mendukung penambahan label baru

### 2. Detection & Segmentation
- Menggunakan model detection untuk menemukan ikan dalam image
- Segmentation untuk mendapatkan mask yang presisi
- Output berupa bounding box + segmentation mask

### 3. Database Management
- Automatic backup sebelum modifikasi
- Validasi image files
- Update incremental database

### 4. Batch Processing
- Tambah multiple species sekaligus
- Progress tracking
- Error handling dan recovery

## Contoh Penggunaan Programmatik

```python
from add_new_fish_dataset import FishDatasetManager

# Initialize
MODEL_DIRS = {
    'classification': "models/classification",
    'segmentation': "models/segmentation", 
    'detection': "models/detection"
}

fish_manager = FishDatasetManager(MODEL_DIRS)

# Tambah spesies baru
fish_manager.add_new_fish_species(
    "dataset/ikan_db_v1/images/mujair", 
    "Mujair"
)

# Save database
fish_manager.save_updated_database()

# Prediksi image baru
results, viz_img = fish_manager.predict_fish("test_image.jpg")
fish_manager.display_results(results, viz_img)
```

## Utilities

### List Available Species
```python
from dataset_utilities import DatasetUtilities

species_info = DatasetUtilities.list_available_species()
DatasetUtilities.display_species_info(species_info)
```

### Check Current Labels
```python
labels = DatasetUtilities.check_current_labels()
```

### Backup Database
```python
backup_dir = DatasetUtilities.backup_current_database()
```

## Output

Sistem akan menghasilkan:

1. **Detection Results**: Bounding box lokasi ikan
2. **Classification Results**: Top-K prediksi spesies dengan confidence score
3. **Segmentation Results**: Mask polygon untuk setiap ikan
4. **Visualization**: Image dengan annotation lengkap

Contoh output classification:
```
Fish 0 Classification Results:
----------------------------------------
1. Mujair
   Species ID: 156
   Accuracy: 0.892
   Distance: 0.234

2. Bandeng  
   Species ID: 89
   Accuracy: 0.756
   Distance: 0.456
```

## Troubleshooting

### Model tidak ditemukan
```bash
# Download models terlebih dahulu
python research_fishial.py
```

### Dataset path tidak valid
- Pastikan path menuju folder yang berisi image files
- Supported format: JPG, JPEG, PNG, BMP

### Memory error
- Kurangi batch size dalam konfigurasi
- Process species satu per satu

### Database corrupted
- Restore dari backup folder yang otomatis dibuat
- Re-run script dengan clean database

## Tips Optimisasi

1. **Image Quality**: Gunakan image dengan resolusi minimal 224x224
2. **Dataset Size**: Minimal 5-10 image per spesies untuk hasil yang baik
3. **Image Variety**: Sertakan variasi pose, lighting, dan background
4. **Backup**: Selalu backup database sebelum modifikasi besar

## Monitoring Performance

Script akan menampilkan:
- Waktu processing per image
- Jumlah embedding yang berhasil diekstrak
- Success rate detection
- Memory usage information

## Extend Functionality

Untuk menambah fitur baru:

1. **Custom Distance Metric**: Modify similarity calculation dalam `EmbeddingClassifier`
2. **Feature Augmentation**: Tambah preprocessing steps
3. **Multi-scale Detection**: Modify detection parameters
4. **Custom Visualization**: Extend display functions

## Dependencies

- OpenCV (cv2)
- PyTorch
- NumPy
- Matplotlib
- PIL
- scikit-learn

Pastikan semua dependencies terinstall:
```bash
pip install -r requirements.txt
```