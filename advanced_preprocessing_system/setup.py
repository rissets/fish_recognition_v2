#!/usr/bin/env python3
"""
🚀 Setup Script for Advanced Fish Preprocessing System
"""

import os
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_directory_structure():
    """Create necessary directory structure"""
    directories = [
        "data",
        "data/sample_fish",
        "data/sample_fish/bandeng",
        "data/sample_fish/mujair", 
        "data/sample_fish/nila",
        "output",
        "output/single_test",
        "output/dataset_processed",
        "output/quality_test",
        "output/full_pipeline_test",
        "models/pretrained"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"📁 Created directory: {directory}")

def copy_sample_images():
    """Copy sample images for testing"""
    # Source images from parent directory
    source_base = "../images"
    dest_base = "data/sample_fish"
    
    sample_mappings = {
        "bandeng": ["bandeng.jpg", "bandeng1.jpeg", "bandeng2.jpg", "bandeng3.jpg"],
        "mujair": ["mujair1.jpg", "mujair2.jpg", "mujair3.jpg"],
        "nila": []  # Add nila images if available
    }
    
    total_copied = 0
    
    for species, filenames in sample_mappings.items():
        species_dir = os.path.join(dest_base, species)
        
        for i, filename in enumerate(filenames):
            source_path = os.path.join(source_base, filename)
            if os.path.exists(source_path):
                dest_filename = f"{species}_{i+1:02d}.jpg"
                dest_path = os.path.join(species_dir, dest_filename)
                shutil.copy2(source_path, dest_path)
                logger.info(f"📋 Copied: {filename} → {dest_filename}")
                total_copied += 1
    
    logger.info(f"✅ Copied {total_copied} sample images")
    return total_copied

def create_example_config():
    """Create example configuration file"""
    config = {
        "preprocessing": {
            "target_size": 512,
            "detection_model": "yolov8n.pt",
            "strategies_count": 10,
            "output_quality": 95
        },
        "training": {
            "batch_size": 16,
            "learning_rate": 0.001,
            "epochs": 10,
            "val_split": 0.2,
            "test_split": 0.2
        },
        "paths": {
            "dataset": "data",
            "output": "output",
            "models": "models"
        }
    }
    
    import json
    with open("config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    logger.info("⚙️  Created configuration file: config.json")

def main():
    """Setup the system"""
    logger.info("🚀 Setting up Advanced Fish Preprocessing System...")
    
    # Create directories
    create_directory_structure()
    
    # Copy sample images
    copied_images = copy_sample_images()
    
    # Create config file
    create_example_config()
    
    # Create README
    readme_content = """# Advanced Fish Preprocessing System

## 🎯 Overview
This system transforms 1 input fish image into 10 enhanced versions using:
- Fish detection and segmentation
- Advanced image augmentation
- Quality enhancement techniques

## 📁 Directory Structure
```
advanced_preprocessing_system/
├── data/                          # Input dataset
│   └── sample_fish/              # Sample fish images
│       ├── bandeng/              # Bandeng species folder
│       ├── mujair/               # Mujair species folder
│       └── nila/                 # Nila species folder
├── output/                       # Processed outputs
├── models/                       # Model files
├── advanced_preprocessing.py     # Main preprocessing system
├── test_system.py               # Testing and evaluation
├── setup.py                    # Setup script
└── requirements.txt            # Dependencies

```

## 🚀 Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run setup:
```bash
python setup.py
```

3. Test preprocessing:
```bash
python advanced_preprocessing.py
```

4. Run full pipeline test:
```bash
python test_system.py
```

## 📊 Features

### Preprocessing (1 → 10 images):
1. **original_enhanced** - Enhanced version of original
2. **bright_contrast** - Brightness and contrast optimization
3. **sharp_saturation** - Sharpness and saturation enhancement
4. **clahe_gamma** - CLAHE and gamma correction
5. **histogram_eq** - Histogram equalization
6. **blur_denoise** - Blur and denoising
7. **edge_enhance** - Edge enhancement
8. **color_balance** - Color balance and auto-contrast
9. **lighting_fix** - Lighting correction
10. **detail_enhance** - Detail enhancement with unsharp mask

### Testing Features:
- Dataset processing from folder structure
- Automatic train/val/test split
- CNN model training and evaluation
- Comprehensive metrics and reports
- Quality assessment of preprocessing

## 🐟 Dataset Format
```
data/
├── species1/          # Folder name = species label
│   ├── image1.jpg    # Images in folder
│   └── image2.jpg
└── species2/
    ├── image1.jpg
    └── image2.jpg
```

## 📈 Results
The system will generate:
- 10 enhanced versions per input image
- Processing reports in JSON format
- Model performance metrics
- Quality assessment results
"""
    
    with open("README.md", "w") as f:
        f.write(readme_content)
    
    logger.info("📖 Created README.md")
    
    # Final summary
    logger.info("\n" + "="*50)
    logger.info("✅ Setup completed successfully!")
    logger.info(f"📊 Sample images copied: {copied_images}")
    logger.info("🎯 Next steps:")
    logger.info("   1. pip install -r requirements.txt")
    logger.info("   2. python advanced_preprocessing.py")
    logger.info("   3. python test_system.py")
    logger.info("="*50)

if __name__ == "__main__":
    main()