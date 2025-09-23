#!/usr/bin/env python3
"""
🧪 Advanced Fish Recognition Testing System
Tests the preprocessing pipeline with various datasets and evaluation metrics
Uses existing classification model for validation
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import logging
import sys

# Add parent directory to path for model imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from advanced_preprocessing import AdvancedFishPreprocessor, FishDatasetProcessor

# Import existing classification model
try:
    from models.classification.inference import EmbeddingClassifier
    EXISTING_CLASSIFIER_AVAILABLE = True
except ImportError as e:
    EXISTING_CLASSIFIER_AVAILABLE = False
    print(f"Warning: Could not import existing classifier: {e}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FishDataset(Dataset):
    """🐟 Custom dataset for preprocessed fish images"""
    
    def __init__(self, data_paths: List[str], labels: List[int], transform=None):
        self.data_paths = data_paths
        self.labels = labels
        self.transform = transform
        self.label_names = {}
    
    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, idx):
        # Load image
        if isinstance(self.data_paths[idx], str):
            image = cv2.imread(self.data_paths[idx])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            # Already loaded image array
            image = self.data_paths[idx]
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, self.labels[idx]


class SimpleCNN(nn.Module):
    """🧠 Simple CNN for fish classification testing"""
    
    def __init__(self, num_classes: int, input_size: int = 224):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Calculate the size after convolutions
        self.feature_size = self._get_conv_output_size(input_size)
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.feature_size, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )
    
    def _get_conv_output_size(self, input_size):
        """Calculate output size after convolutions"""
        x = torch.randn(1, 3, input_size, input_size)
        x = self.features(x)
        return x.view(1, -1).size(1)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ExistingModelTester:
    """🔬 Tester using existing classification model"""
    
    def __init__(self):
        self.models_base = os.path.join(os.path.dirname(__file__), '..', 'models')
        self.classification_model_path = os.path.join(self.models_base, 'classification', 'model.ts')
        self.database_path = os.path.join(self.models_base, 'classification', 'database.pt')
        self.labels_path = os.path.join(self.models_base, 'classification', 'labels.json')
        
        self.classifier = None
        self.labels_map = {}
        self.load_existing_model()
        
        self.preprocessor = AdvancedFishPreprocessor()
    
    def load_existing_model(self):
        """Load existing classification model"""
        if not EXISTING_CLASSIFIER_AVAILABLE:
            logger.warning("⚠️ Existing classifier not available")
            return
            
        if os.path.exists(self.classification_model_path) and os.path.exists(self.database_path):
            try:
                self.classifier = EmbeddingClassifier(
                    model_path=self.classification_model_path,
                    data_set_path=self.database_path,
                    device='cpu',
                    labels_file=self.labels_path
                )
                logger.info("✅ Existing classification model loaded successfully")
                
                # Load labels mapping
                if os.path.exists(self.labels_path):
                    with open(self.labels_path, 'r', encoding='utf-8') as f:
                        self.labels_map = json.load(f)
                        logger.info(f"📋 Loaded {len(self.labels_map)} fish species labels")
                        
            except Exception as e:
                logger.error(f"❌ Failed to load existing model: {e}")
                self.classifier = None
        else:
            logger.warning("⚠️ Existing model files not found")
    
    def test_preprocessing_with_classification(self, image_path: str) -> Dict:
        """Test preprocessing pipeline with existing classifier"""
        if self.classifier is None:
            return {"error": "Classifier not available"}
        
        try:
            # Load original image
            image = cv2.imread(image_path)
            if image is None:
                return {"error": f"Could not load image: {image_path}"}
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process with our preprocessing system
            results = self.preprocessor.process_single_image(
                image_path, 
                output_dir="output/quality_test"
            )
            
            if not results:
                return {"error": "Preprocessing failed"}
            
            # Test classification on each processed version
            classification_results = {}
            
            for result in results:
                strategy = result.get("strategy_name", "unknown")
                processed_path = result.get("output_path")
                
                if not processed_path or not os.path.exists(processed_path):
                    continue
                
                try:
                    # Load processed image
                    processed_img = cv2.imread(processed_path)
                    if processed_img is None:
                        continue
                    
                    processed_img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                    
                    # Use existing classifier to classify processed image
                    predictions = self.classifier.inference_numpy(processed_img_rgb)
                    
                    if predictions and len(predictions) > 0:
                        top_prediction = predictions[0]
                        
                        classification_results[strategy] = {
                            "confidence": float(top_prediction.get('accuracy', 0)),
                            "species": top_prediction.get('name', 'unknown'),
                            "species_id": top_prediction.get('species_id', None),
                            "processed_image": processed_path
                        }
                    else:
                        classification_results[strategy] = {
                            "confidence": 0.0,
                            "species": "no_prediction",
                            "processed_image": processed_path
                        }
                        
                except Exception as e:
                    logger.warning(f"Classification failed for {strategy}: {e}")
                    classification_results[strategy] = {
                        "error": str(e),
                        "processed_image": processed_path
                    }
            
            # Analyze results
            analysis = self.analyze_preprocessing_quality(classification_results)
            
            return {
                "original_image": image_path,
                "total_processed": len(results),
                "classifications": classification_results,
                "analysis": analysis
            }
            
        except Exception as e:
            logger.error(f"Testing failed: {e}")
            return {"error": str(e)}
    
    def analyze_preprocessing_quality(self, results: Dict) -> Dict:
        """Analyze quality of preprocessing based on classification results"""
        valid_results = {k: v for k, v in results.items() if "error" not in v}
        
        if not valid_results:
            return {"error": "No valid classification results"}
        
        # Calculate metrics
        confidences = [v["confidence"] for v in valid_results.values()]
        species_counts = {}
        
        for result in valid_results.values():
            species = result["species"]
            species_counts[species] = species_counts.get(species, 0) + 1
        
        # Find most predicted species
        most_common_species = max(species_counts.items(), key=lambda x: x[1]) if species_counts else ("unknown", 0)
        
        # Quality metrics
        avg_confidence = np.mean(confidences) if confidences else 0
        max_confidence = max(confidences) if confidences else 0
        min_confidence = min(confidences) if confidences else 0
        consistency = most_common_species[1] / len(valid_results) if valid_results else 0
        
        return {
            "average_confidence": float(avg_confidence),
            "max_confidence": float(max_confidence), 
            "min_confidence": float(min_confidence),
            "consistency_ratio": float(consistency),
            "most_predicted_species": most_common_species[0],
            "species_distribution": species_counts,
            "processing_strategies_tested": len(valid_results),
            "quality_score": float((avg_confidence + consistency) / 2)
        }


class FishRecognitionTester:
    """🧪 Comprehensive testing system for fish recognition"""
    
    def __init__(self, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🖥️  Using device: {self.device}")
        
        self.preprocessor = AdvancedFishPreprocessor()
        self.processor = FishDatasetProcessor(self.preprocessor)
        
        # Training transforms
        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Test transforms
        self.test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def prepare_dataset_from_processing_results(self, processing_results: Dict) -> Tuple[List, List, Dict]:
        """Prepare dataset from preprocessing results"""
        data_paths = []
        labels = []
        label_to_idx = {}
        idx_to_label = {}
        
        current_label_idx = 0
        
        for species_name, species_data in processing_results['species'].items():
            if species_name not in label_to_idx:
                label_to_idx[species_name] = current_label_idx
                idx_to_label[current_label_idx] = species_name
                current_label_idx += 1
            
            species_label = label_to_idx[species_name]
            
            # Add all processed versions for this species
            for image_data in species_data['images']:
                for version in image_data['processed_versions']:
                    if version['output_path'] and os.path.exists(version['output_path']):
                        data_paths.append(version['output_path'])
                        labels.append(species_label)
        
        logger.info(f"📊 Dataset prepared: {len(data_paths)} samples, {len(label_to_idx)} classes")
        return data_paths, labels, {'label_to_idx': label_to_idx, 'idx_to_label': idx_to_label}
    
    def train_simple_model(self, train_loader: DataLoader, val_loader: DataLoader, 
                          num_classes: int, epochs: int = 10) -> SimpleCNN:
        """Train a simple CNN model"""
        model = SimpleCNN(num_classes).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        
        logger.info(f"🏋️ Training model for {epochs} epochs...")
        
        train_losses = []
        val_accuracies = []
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
            
            train_acc = 100. * correct / total
            avg_loss = total_loss / len(train_loader)
            train_losses.append(avg_loss)
            
            # Validation phase
            val_acc = self.evaluate_model(model, val_loader)
            val_accuracies.append(val_acc)
            
            scheduler.step()
            
            logger.info(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")
        
        return model
    
    def evaluate_model(self, model: SimpleCNN, test_loader: DataLoader) -> float:
        """Evaluate model on test data"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        accuracy = 100. * correct / total
        return accuracy
    
    def detailed_evaluation(self, model: SimpleCNN, test_loader: DataLoader, 
                          idx_to_label: Dict) -> Dict:
        """Detailed evaluation with metrics"""
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                _, predicted = output.max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # Convert to class names
        pred_names = [idx_to_label[p] for p in all_predictions]
        true_names = [idx_to_label[t] for t in all_targets]
        
        # Calculate metrics
        accuracy = sum(p == t for p, t in zip(all_predictions, all_targets)) / len(all_targets)
        
        # Classification report
        class_report = classification_report(true_names, pred_names, output_dict=True)
        
        # Confusion matrix
        conf_matrix = confusion_matrix(true_names, pred_names)
        
        return {
            'accuracy': accuracy,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix,
            'predictions': pred_names,
            'true_labels': true_names,
            'class_names': list(idx_to_label.values())
        }
    
    def test_preprocessing_quality(self, dataset_path: str, output_path: str = "output/quality_test") -> Dict:
        """Test preprocessing quality metrics"""
        logger.info("🔍 Testing preprocessing quality...")
        
        # Process dataset
        results = self.processor.process_dataset(dataset_path, output_path)
        
        quality_metrics = {
            'total_images_processed': results['total_images'],
            'total_versions_generated': results['total_processed'],
            'success_rate': (results['total_processed'] / (results['total_images'] * 10)) * 100,
            'failed_images': len(results['failed_images']),
            'species_coverage': len(results['species']),
            'average_versions_per_image': results['total_processed'] / results['total_images'] if results['total_images'] > 0 else 0
        }
        
        logger.info(f"📊 Preprocessing Quality Metrics:")
        logger.info(f"   Success Rate: {quality_metrics['success_rate']:.1f}%")
        logger.info(f"   Versions per Image: {quality_metrics['average_versions_per_image']:.1f}")
        logger.info(f"   Species Covered: {quality_metrics['species_coverage']}")
        
        return quality_metrics, results
    
    def run_full_pipeline_test(self, dataset_path: str, output_path: str = "output/full_test", 
                              epochs: int = 5) -> Dict:
        """Run complete pipeline test"""
        logger.info("🚀 Running full pipeline test...")
        
        # Step 1: Test preprocessing
        quality_metrics, processing_results = self.test_preprocessing_quality(dataset_path, output_path)
        
        if processing_results['total_processed'] == 0:
            logger.error("❌ No images were processed successfully")
            return {"error": "No processed images"}
        
        # Step 2: Prepare dataset
        data_paths, labels, label_info = self.prepare_dataset_from_processing_results(processing_results)
        
        if len(data_paths) < 10:  # Minimum samples for meaningful test
            logger.warning("⚠️ Too few samples for training, skipping model test")
            return {
                "preprocessing_quality": quality_metrics,
                "dataset_info": {"total_samples": len(data_paths), "num_classes": len(label_info['label_to_idx'])},
                "model_test": "skipped_insufficient_data"
            }
        
        # Step 3: Split dataset
        train_paths, test_paths, train_labels, test_labels = train_test_split(
            data_paths, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_paths, train_labels, test_size=0.2, random_state=42, stratify=train_labels
        )
        
        # Step 4: Create datasets and loaders
        train_dataset = FishDataset(train_paths, train_labels, self.train_transform)
        val_dataset = FishDataset(val_paths, val_labels, self.test_transform)
        test_dataset = FishDataset(test_paths, test_labels, self.test_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        # Step 5: Train model
        num_classes = len(label_info['label_to_idx'])
        model = self.train_simple_model(train_loader, val_loader, num_classes, epochs)
        
        # Step 6: Evaluate model
        evaluation_results = self.detailed_evaluation(model, test_loader, label_info['idx_to_label'])
        
        # Step 7: Compile final results
        final_results = {
            "preprocessing_quality": quality_metrics,
            "dataset_info": {
                "total_samples": len(data_paths),
                "num_classes": num_classes,
                "class_names": list(label_info['label_to_idx'].keys()),
                "train_samples": len(train_paths),
                "val_samples": len(val_paths),
                "test_samples": len(test_paths)
            },
            "model_performance": {
                "test_accuracy": evaluation_results['accuracy'] * 100,
                "classification_report": evaluation_results['classification_report']
            }
        }
        
        # Save results
        results_path = os.path.join(output_path, "full_test_results.json")
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        logger.info(f"✅ Full pipeline test completed!")
        logger.info(f"   Test Accuracy: {final_results['model_performance']['test_accuracy']:.2f}%")
        logger.info(f"   Results saved: {results_path}")
        
        return final_results


def create_sample_dataset():
    """Create sample dataset structure for testing"""
    sample_data_path = "data/sample_fish"
    os.makedirs(sample_data_path, exist_ok=True)
    
    # Create sample species directories
    species = ["bandeng", "mujair", "nila"]
    
    for species_name in species:
        species_dir = os.path.join(sample_data_path, species_name)
        os.makedirs(species_dir, exist_ok=True)
        
        # Copy sample images if they exist
        source_images = [
            f"../images/{species_name}.jpg",
            f"../images/{species_name}1.jpg",
            f"../images/{species_name}2.jpg"
        ]
        
        for i, source_img in enumerate(source_images):
            if os.path.exists(source_img):
                import shutil
                dest_img = os.path.join(species_dir, f"{species_name}_{i+1}.jpg")
                shutil.copy2(source_img, dest_img)
                logger.info(f"📋 Copied: {source_img} → {dest_img}")
    
    return sample_data_path


def main():
    """🧪 Main testing function"""
    print("🧪 Fish Recognition Advanced Testing System")
    print("=" * 50)
    
    # Initialize tester
    tester = FishRecognitionTester()
    
    # Create sample dataset if no data directory exists
    if not os.path.exists("data") or not any(os.path.isdir(os.path.join("data", d)) for d in os.listdir("data")):
        print("📋 Creating sample dataset...")
        sample_dataset = create_sample_dataset()
        if os.path.exists(sample_dataset):
            dataset_path = sample_dataset
        else:
            dataset_path = "data"
    else:
        dataset_path = "data"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset path not found: {dataset_path}")
        print("Please create the directory structure:")
        print("data/")
        print("  ├── species1/")
        print("  │   ├── image1.jpg")
        print("  │   └── image2.jpg")
        print("  └── species2/")
        print("      ├── image1.jpg")
        print("      └── image2.jpg")
        return
    
    # Run tests
    try:
        print(f"\n🚀 Running full pipeline test on: {dataset_path}")
        start_time = time.time()
        
        results = tester.run_full_pipeline_test(dataset_path, "output/full_pipeline_test", epochs=3)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n⏱️  Total test time: {total_time:.2f} seconds")
        print(f"📊 Final Results Summary:")
        
        if "error" not in results:
            print(f"   Preprocessing Success Rate: {results['preprocessing_quality']['success_rate']:.1f}%")
            print(f"   Total Processed Images: {results['preprocessing_quality']['total_versions_generated']}")
            print(f"   Species Detected: {results['dataset_info']['num_classes']}")
            
            if "model_performance" in results:
                print(f"   Model Test Accuracy: {results['model_performance']['test_accuracy']:.2f}%")
            else:
                print(f"   Model Test: {results.get('model_test', 'Not performed')}")
        else:
            print(f"   Error: {results['error']}")
        
        print(f"\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()