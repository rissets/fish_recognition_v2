#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Fish Recognition System
Implementasi lengkap untuk menambah ikan ke database secara permanen
dengan mempertahankan struktur database yang benar
"""

import os
import cv2
import torch
import numpy as np
import json
import uuid
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import model inference
import sys
sys.path.append('..')
from models.classification.inference import EmbeddingClassifier
from models.detection.inference import YOLOInference
from models.segmentation.inference import Inference

class DatabaseManager:
    """Manager untuk mengelola database embedding dengan struktur yang benar"""
    
    def __init__(self, db_path: str, labels_path: str):
        self.db_path = db_path
        self.labels_path = labels_path
        self.backup_dir = Path(db_path).parent / "backup"
        self.backup_dir.mkdir(exist_ok=True)
        
    def create_backup(self) -> str:
        """Buat backup database sebelum modifikasi"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"database_backup_{timestamp}.pt"
        
        if os.path.exists(self.db_path):
            shutil.copy2(self.db_path, backup_path)
            logger.info(f"Database backup created: {backup_path}")
            return str(backup_path)
        return ""
    
    def load_database(self) -> Tuple[torch.Tensor, List, List, List, List, Dict]:
        """Load database dengan validasi struktur"""
        try:
            if not os.path.exists(self.db_path):
                logger.warning(f"Database not found: {self.db_path}")
                return self._create_empty_database()
            
            db_data = torch.load(self.db_path)
            
            # Validasi struktur database (harus 6 elemen)
            if not isinstance(db_data, (list, tuple)) or len(db_data) != 6:
                logger.error(f"Invalid database structure. Expected 6 elements, got {len(db_data) if hasattr(db_data, '__len__') else 'unknown'}")
                return self._create_empty_database()
            
            embeddings = db_data[0]
            internal_ids = db_data[1]
            image_ids = db_data[2]
            annotation_ids = db_data[3]
            drawn_fish_ids = db_data[4]
            keys = db_data[5]
            
            # Validasi konsistensi panjang data
            lengths = [
                embeddings.shape[0] if isinstance(embeddings, torch.Tensor) else 0,
                len(internal_ids),
                len(image_ids),
                len(annotation_ids),
                len(drawn_fish_ids)
            ]
            
            if len(set(lengths)) > 1:
                logger.warning(f"Inconsistent data lengths: {lengths}")
                min_len = min(lengths)
                logger.info(f"Truncating to minimum length: {min_len}")
                
                # Truncate semua ke panjang minimum
                if isinstance(embeddings, torch.Tensor) and embeddings.shape[0] > min_len:
                    embeddings = embeddings[:min_len]
                internal_ids = internal_ids[:min_len]
                image_ids = image_ids[:min_len]
                annotation_ids = annotation_ids[:min_len]
                drawn_fish_ids = drawn_fish_ids[:min_len]
            
            logger.info(f"Database loaded successfully with {embeddings.shape[0] if isinstance(embeddings, torch.Tensor) else 0} entries")
            return embeddings, internal_ids, image_ids, annotation_ids, drawn_fish_ids, keys
            
        except Exception as e:
            logger.error(f"Error loading database: {e}")
            return self._create_empty_database()
    
    def _create_empty_database(self) -> Tuple[torch.Tensor, List, List, List, List, Dict]:
        """Buat database kosong dengan struktur yang benar"""
        logger.info("Creating empty database structure")
        empty_embeddings = torch.empty(0, 128)  # Assuming 128-dim embeddings
        return (
            empty_embeddings,
            [],  # internal_ids
            [],  # image_ids
            [],  # annotation_ids
            [],  # drawn_fish_ids
            {}   # keys
        )
    
    def save_database(self, embeddings: torch.Tensor, internal_ids: List, 
                     image_ids: List, annotation_ids: List, 
                     drawn_fish_ids: List, keys: Dict) -> bool:
        """Simpan database dengan validasi"""
        try:
            # Validasi panjang data
            lengths = [embeddings.shape[0], len(internal_ids), len(image_ids), 
                      len(annotation_ids), len(drawn_fish_ids)]
            
            if len(set(lengths)) > 1:
                logger.error(f"Data length mismatch: {lengths}")
                return False
            
            # Buat backup sebelum save
            self.create_backup()
            
            # Save database
            db_data = [embeddings, internal_ids, image_ids, annotation_ids, drawn_fish_ids, keys]
            torch.save(db_data, self.db_path)
            
            logger.info(f"Database saved successfully with {embeddings.shape[0]} entries")
            return True
            
        except Exception as e:
            logger.error(f"Error saving database: {e}")
            return False
    
    def load_labels(self) -> Dict[str, str]:
        """Load labels dari file JSON"""
        try:
            if os.path.exists(self.labels_path):
                with open(self.labels_path, 'r', encoding='utf-8') as f:
                    labels = json.load(f)
                logger.info(f"Loaded {len(labels)} labels")
                return labels
            else:
                logger.warning(f"Labels file not found: {self.labels_path}")
                return {}
        except Exception as e:
            logger.error(f"Error loading labels: {e}")
            return {}
    
    def save_labels(self, labels: Dict[str, str]) -> bool:
        """Simpan labels ke file JSON"""
        try:
            with open(self.labels_path, 'w', encoding='utf-8') as f:
                json.dump(labels, f, ensure_ascii=False, indent=2)
            logger.info(f"Labels saved successfully with {len(labels)} entries")
            return True
        except Exception as e:
            logger.error(f"Error saving labels: {e}")
            return False
    
    def add_new_species_label(self, species_name: str) -> int:
        """Tambahkan spesies baru ke labels dan return ID-nya"""
        labels = self.load_labels()
        
        # Cek apakah spesies sudah ada
        for label_id, label_name in labels.items():
            if label_name.lower() == species_name.lower():
                logger.info(f"Species '{species_name}' already exists with ID {label_id}")
                return int(label_id)
        
        # Tambah spesies baru
        max_id = max([int(k) for k in labels.keys()], default=-1)
        new_id = max_id + 1
        labels[str(new_id)] = species_name
        
        if self.save_labels(labels):
            logger.info(f"Added new species '{species_name}' with ID {new_id}")
            return new_id
        else:
            logger.error(f"Failed to save new species '{species_name}'")
            return -1

class AdvancedFishSystem:
    """Advanced Fish Recognition System dengan database management yang robust"""
    
    def __init__(self, base_dir: str = ".."):
        self.base_dir = Path(base_dir)
        self.initialized = False
        
        # Path model
        self.classifier_model_path = self.base_dir / 'models/classification/model.ts'
        self.classifier_db_path = self.base_dir / 'models/classification/database.pt'
        self.detector_model_path = self.base_dir / 'models/detection/model.ts'
        self.segmentation_model_path = self.base_dir / 'models/segmentation/model.ts'
        self.labels_path = self.base_dir / 'models/classification/labels.json'
        
        # Initialize database manager
        self.db_manager = DatabaseManager(str(self.classifier_db_path), str(self.labels_path))
        
        # Initialize models
        try:
            self._load_models()
        except Exception as e:
            logger.error(f"Failed to initialize system: {e}")
            self.initialized = False
    
    def _load_models(self):
        """Load semua model yang diperlukan"""
        try:
            logger.info("Loading models...")
            
            self.detector = YOLOInference(
                str(self.detector_model_path),
                imsz=(640, 640),
                conf_threshold=0.7,
                nms_threshold=0.3,
                yolo_ver='v10'
            )
            logger.info("✅ Detector loaded")
            
            self.segmentator = Inference(
                model_path=str(self.segmentation_model_path),
                image_size=416
            )
            logger.info("✅ Segmentator loaded")
            
            self.classifier = EmbeddingClassifier(
                str(self.classifier_model_path),
                str(self.classifier_db_path),
                labels_file=str(self.labels_path)
            )
            logger.info("✅ Classifier loaded")
            
            logger.info("🎉 All models loaded successfully!")
            self.initialized = True
            
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
            self.initialized = False
    
    def reload_classifier(self):
        """Reload classifier setelah database diupdate"""
        try:
            self.classifier = EmbeddingClassifier(
                str(self.classifier_model_path),
                str(self.classifier_db_path),
                labels_file=str(self.labels_path)
            )
            logger.info("✅ Classifier reloaded with updated database")
        except Exception as e:
            logger.error(f"❌ Error reloading classifier: {e}")
            self.initialized = False
            return True
        except Exception as e:
            logger.error(f"❌ Error reloading classifier: {e}")
            return False
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Dapatkan statistik database"""
        embeddings, internal_ids, image_ids, annotation_ids, drawn_fish_ids, keys = self.db_manager.load_database()
        labels = self.db_manager.load_labels()
        
        return {
            'total_embeddings': embeddings.shape[0] if isinstance(embeddings, torch.Tensor) else 0,
            'embedding_dimension': embeddings.shape[1] if isinstance(embeddings, torch.Tensor) and embeddings.numel() > 0 else 'N/A',
            'total_internal_ids': len(internal_ids),
            'total_image_ids': len(image_ids),
            'total_annotation_ids': len(annotation_ids),
            'total_drawn_fish_ids': len(drawn_fish_ids),
            'total_keys': len(keys),
            'total_labels': len(labels),
            'available_species': labels
        }
    
    def predict_fish_complete(self, image_path: str) -> Optional[Dict[str, Any]]:
        """
        Prediksi lengkap ikan: detection + classification + segmentation
        
        Args:
            image_path: Path ke gambar ikan
            
        Returns:
            Dict dengan hasil prediksi atau None jika gagal
        """
        try:
            if not self.initialized:
                logger.error("System not initialized")
                return None
            
            if not os.path.exists(image_path):
                logger.error(f"Image not found: {image_path}")
                return None
            
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Failed to load image: {image_path}")
                return None
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 1. Detection
            detection_results = self.detector.predict(img_rgb)
            
            if not detection_results or len(detection_results) == 0 or len(detection_results[0]) == 0:
                logger.warning("No fish detected in image")
                return {
                    'species': 'No fish detected',
                    'confidence': 0.0,
                    'detection_confidence': 0.0,
                    'has_mask': False,
                    'bbox': None,
                    'mask': None
                }
            
            # Get best detection
            best_detection = detection_results[0][0]
            detection_confidence = float(best_detection.score)
            
            # Get bounding box
            bbox = [
                int(best_detection.x1),
                int(best_detection.y1),
                int(best_detection.x2),
                int(best_detection.y2)
            ]
            
            # Crop detected fish for classification
            cropped_bgr = best_detection.get_mask_BGR()
            
            # 2. Classification
            try:
                # Check if database has data
                db_data = torch.load(str(self.classifier_db_path))
                embeddings = db_data[0]
                
                if embeddings.numel() == 0:
                    species = "Unknown"
                    confidence = 0.0
                else:
                    classification_results = self.classifier.batch_inference([cropped_bgr])
                    
                    if classification_results and len(classification_results) > 0 and classification_results[0]:
                        top_result = classification_results[0][0]
                        species = top_result['name']
                        confidence = top_result['accuracy']
                    else:
                        species = "Unknown"
                        confidence = 0.0
                        
            except Exception as e:
                logger.warning(f"Classification failed: {e}")
                species = "Unknown"
                confidence = 0.0
            
            # 3. Segmentation
            has_mask = False
            mask = None
            
            try:
                # Get segmentation mask
                segmentation_result = self.segmentator.predict(img_rgb)
                if segmentation_result is not None:
                    has_mask = True
                    mask = segmentation_result
            except Exception as e:
                logger.warning(f"Segmentation failed: {e}")
            
            result = {
                'species': species,
                'confidence': float(confidence),
                'detection_confidence': detection_confidence,
                'has_mask': has_mask,
                'bbox': bbox,
                'mask': mask
            }
            
            logger.info(f"Prediction completed: {species} (conf: {confidence:.3f})")
            return result
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return None
    
    def add_fish_to_database_permanent(self, image_path: str, species_name: str, 
                                     annotation_source: str = "manual") -> bool:
        """
        Menambahkan ikan baru ke database secara permanen
        
        Args:
            image_path: Path ke gambar ikan
            species_name: Nama spesies ikan
            annotation_source: Sumber anotasi (manual, auto, etc.)
        
        Returns:
            bool: True jika berhasil, False jika gagal
        """
        logger.info(f"🔄 Adding fish to database permanently: {species_name} from {image_path}")
        
        try:
            # 1. Validasi input
            if not os.path.exists(image_path):
                logger.error(f"❌ Image not found: {image_path}")
                return False
            
            img_bgr = cv2.imread(image_path)
            if img_bgr is None:
                logger.error(f"❌ Cannot read image: {image_path}")
                return False
            
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            # 2. Detection
            logger.info("   🔍 Detecting fish...")
            detection_results = self.detector.predict(img_rgb)
            if not detection_results or len(detection_results[0]) == 0:
                logger.error("❌ No fish detected in image!")
                return False
            
            # Ambil deteksi terbaik
            box = detection_results[0][0]
            cropped_bgr = box.get_mask_BGR()
            logger.info(f"   ✅ Fish detected with confidence: {box.score:.3f}")
            
            # 3. Generate embedding
            logger.info("   🧠 Generating embedding...")
            from PIL import Image
            image = Image.fromarray(cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB))
            image_tensor = self.classifier.loader(image).unsqueeze(0).to('cpu')
            
            with torch.no_grad():
                embedding, _ = self.classifier.model(image_tensor)
                embedding = embedding[0]  # Ambil embedding pertama dari batch
            
            logger.info(f"   📊 Embedding generated - shape: {embedding.shape}, norm: {torch.norm(embedding).item():.4f}")
            
            # 4. Load database existing
            embeddings, internal_ids, image_ids, annotation_ids, drawn_fish_ids, keys = self.db_manager.load_database()
            
            # 5. Generate IDs untuk entry baru
            current_time = datetime.now()
            
            # Generate unique IDs
            new_internal_id = len(internal_ids)  # Sequential internal ID
            new_image_id = str(uuid.uuid4())  # Unique image ID
            new_annotation_id = str(uuid.uuid4())  # Unique annotation ID
            new_drawn_fish_id = str(uuid.uuid4())  # Unique drawn fish ID
            
            # 6. Add species to labels if not exists
            species_id = self.db_manager.add_new_species_label(species_name)
            if species_id == -1:
                logger.error(f"❌ Failed to add species label: {species_name}")
                return False
            
            # 7. Update database arrays
            # Embeddings - concatenate new embedding
            if embeddings.numel() == 0:
                new_embeddings = embedding.unsqueeze(0)
            else:
                new_embeddings = torch.cat([embeddings, embedding.unsqueeze(0)], dim=0)
            
            # Lists - append new entries
            new_internal_ids = internal_ids + [new_internal_id]
            new_image_ids = image_ids + [new_image_id]
            new_annotation_ids = annotation_ids + [new_annotation_id]
            new_drawn_fish_ids = drawn_fish_ids + [new_drawn_fish_id]
            
            # Keys - update with new species info
            new_keys = keys.copy()
            new_keys[species_id] = {
                'label': species_name,
                'added_date': current_time.isoformat(),
                'source': annotation_source,
                'image_path': image_path
            }
            
            # 8. Save updated database
            success = self.db_manager.save_database(
                new_embeddings, new_internal_ids, new_image_ids,
                new_annotation_ids, new_drawn_fish_ids, new_keys
            )
            
            if not success:
                logger.error("❌ Failed to save database")
                return False
            
            # 9. Reload classifier dengan database baru
            if not self.reload_classifier():
                logger.warning("⚠️ Failed to reload classifier, but database was saved")
            
            logger.info(f"✅ Successfully added {species_name} to database!")
            logger.info(f"   📊 Database now contains {new_embeddings.shape[0]} embeddings")
            logger.info(f"   🆔 Internal ID: {new_internal_id}")
            logger.info(f"   🆔 Image ID: {new_image_id}")
            logger.info(f"   🆔 Species ID: {species_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error adding fish to database: {e}")
            return False
    
    def batch_add_from_folder(self, folder_path: str, species_name: str, 
                             max_images: int = 10) -> Dict[str, int]:
        """
        Batch add multiple fish images from a folder
        
        Args:
            folder_path: Path ke folder berisi gambar
            species_name: Nama spesies
            max_images: Maksimal jumlah gambar yang diproses
        
        Returns:
            Dict dengan statistik hasil
        """
        logger.info(f"🔄 Batch adding from folder: {folder_path}")
        
        if not os.path.exists(folder_path):
            logger.error(f"❌ Folder not found: {folder_path}")
            return {'success': 0, 'failed': 0, 'total': 0}
        
        # Cari semua file gambar
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(folder_path).glob(f"*{ext}"))
        
        if not image_files:
            logger.warning(f"⚠️ No images found in folder: {folder_path}")
            return {'success': 0, 'failed': 0, 'total': 0}
        
        # Process images
        success_count = 0
        failed_count = 0
        total_count = min(len(image_files), max_images)
        
        logger.info(f"📁 Found {len(image_files)} images, processing {total_count}")
        
        for i, image_path in enumerate(image_files[:max_images]):
            logger.info(f"Processing {i+1}/{total_count}: {image_path.name}")
            
            success = self.add_fish_to_database_permanent(
                str(image_path), species_name, annotation_source="batch_auto"
            )
            
            if success:
                success_count += 1
                logger.info(f"   ✅ Success: {image_path.name}")
            else:
                failed_count += 1
                logger.warning(f"   ❌ Failed: {image_path.name}")
        
        results = {
            'success': success_count,
            'failed': failed_count,
            'total': total_count
        }
        
        logger.info(f"🎯 Batch processing completed: {success_count}/{total_count} successful")
        return results
    
    def predict_fish_advanced(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Prediksi ikan dengan hasil yang lebih detail
        """
        logger.info(f"🔮 Advanced prediction for: {image_path}")
        
        try:
            if not os.path.exists(image_path):
                logger.error(f"❌ Image not found: {image_path}")
                return []
            
            img_bgr = cv2.imread(image_path)
            if img_bgr is None:
                logger.error(f"❌ Cannot read image: {image_path}")
                return []
            
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            # Detection
            logger.info("   🔍 Fish Detection...")
            detection_results = self.detector.predict(img_rgb)
            if not detection_results or len(detection_results[0]) == 0:
                logger.warning("⚠️ No fish detected!")
                return []
            
            logger.info(f"✅ Detected {len(detection_results[0])} fish(es)")
            
            results = []
            
            for i, box in enumerate(detection_results[0]):
                logger.info(f"   🐟 Processing Fish #{i+1}:")
                
                # Crop image
                cropped_bgr = box.get_mask_BGR()
                cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
                
                # Classification
                try:
                    classification_results = self.classifier.batch_inference([cropped_bgr])
                    
                    if classification_results and len(classification_results) > 0 and classification_results[0]:
                        top_result = classification_results[0][0]
                        species = top_result['name']
                        accuracy = top_result['accuracy']
                        logger.info(f"      Species: {species} (confidence: {accuracy:.3f})")
                    else:
                        species = "Unknown"
                        accuracy = 0.0
                        logger.info("      Species: Unknown")
                        
                except Exception as e:
                    species = "Unknown"
                    accuracy = 0.0
                    logger.error(f"      Classification error: {e}")
                
                # Segmentation
                try:
                    segmentation_results = self.segmentator.predict(cropped_bgr)
                    if segmentation_results and len(segmentation_results) > 0:
                        mask = segmentation_results[0].mask_polygon(cropped_bgr)
                        seg_success = True
                    else:
                        mask = cropped_rgb
                        seg_success = False
                        
                except Exception as e:
                    mask = cropped_rgb
                    seg_success = False
                    logger.error(f"      Segmentation error: {e}")
                
                # Compile results
                result = {
                    'fish_id': i + 1,
                    'species': species,
                    'classification_confidence': float(accuracy),
                    'detection_confidence': float(box.score),
                    'bounding_box': {
                        'x1': int(box.x1),
                        'y1': int(box.y1),
                        'x2': int(box.x2),
                        'y2': int(box.y2),
                        'width': int(box.x2 - box.x1),
                        'height': int(box.y2 - box.y1)
                    },
                    'segmentation_success': seg_success,
                    'cropped_image': cropped_rgb,
                    'segmentation_mask': mask,
                    'timestamp': datetime.now().isoformat()
                }
                
                results.append(result)
            
            logger.info(f"🎯 Prediction completed for {len(results)} fish(es)")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error in prediction: {e}")
            return []

def main():
    """Demo penggunaan Advanced Fish System"""
    print("🐟 Advanced Fish Recognition System Demo")
    print("=" * 60)
    
    # Initialize system
    system = AdvancedFishSystem()
    
    # Show database stats
    print("\n📊 Current Database Statistics:")
    stats = system.get_database_stats()
    for key, value in stats.items():
        if key != 'available_species':
            print(f"   {key}: {value}")
    
    print(f"\n🏷️ Available species: {stats['total_labels']} species")
    
    # Demo 1: Add single fish
    print("\n" + "="*60)
    print("🎓 DEMO 1: Adding Single Fish to Database")
    print("="*60)
    
    test_image = "../images/mujair1.jpg"
    if os.path.exists(test_image):
        success = system.add_fish_to_database_permanent(test_image, "Mujair")
        if success:
            print("✅ Single fish addition successful!")
        else:
            print("❌ Single fish addition failed!")
    else:
        print(f"⚠️ Test image not found: {test_image}")
    
    # Demo 2: Batch add from folder
    print("\n" + "="*60)
    print("🎓 DEMO 2: Batch Adding from Folder")
    print("="*60)
    
    # Create test folder with sample images
    test_folder = Path("../images")
    if test_folder.exists():
        results = system.batch_add_from_folder(str(test_folder), "Ikan Test", max_images=2)
        print(f"📊 Batch results: {results}")
    else:
        print(f"⚠️ Test folder not found: {test_folder}")
    
    # Demo 3: Advanced prediction
    print("\n" + "="*60)
    print("🔮 DEMO 3: Advanced Prediction")
    print("="*60)
    
    test_prediction_image = "../image_testing/mujair4.jpg"
    if os.path.exists(test_prediction_image):
        prediction_results = system.predict_fish_advanced(test_prediction_image)
        
        print(f"\n📋 Prediction Results:")
        for result in prediction_results:
            print(f"   Fish #{result['fish_id']}:")
            print(f"     Species: {result['species']}")
            print(f"     Classification confidence: {result['classification_confidence']:.3f}")
            print(f"     Detection confidence: {result['detection_confidence']:.3f}")
            print(f"     Segmentation: {'✅' if result['segmentation_success'] else '❌'}")
            print(f"     Bounding box: {result['bounding_box']}")
    else:
        print(f"⚠️ Test image not found: {test_prediction_image}")
    
    # Show final database stats
    print("\n📊 Final Database Statistics:")
    final_stats = system.get_database_stats()
    for key, value in final_stats.items():
        if key != 'available_species':
            print(f"   {key}: {value}")
    
    print("\n🎉 Advanced Fish System Demo Completed!")

if __name__ == "__main__":
    main()