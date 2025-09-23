#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Processing System untuk Advanced Fish Recognition
Batch add multiple species dari struktur folder yang terorganisir
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any
import logging
from dataclasses import dataclass
from datetime import datetime

from advanced_fish_recognition import AdvancedFishSystem
from database_utilities import DatabaseUtilities

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingResult:
    """Class untuk menyimpan hasil processing"""
    species_name: str
    total_images: int
    successful: int
    failed: int
    error_messages: List[str]
    processing_time: float
    
    @property
    def success_rate(self) -> float:
        if self.total_images == 0:
            return 0.0
        return (self.successful / self.total_images) * 100

class BatchProcessor:
    """Class untuk batch processing multiple species"""
    
    def __init__(self, base_dir: str = ".."):
        self.system = AdvancedFishSystem(base_dir)
        self.db_utils = DatabaseUtilities(
            str(Path(base_dir) / "models/classification/database.pt"),
            str(Path(base_dir) / "models/classification/labels.json")
        )
        self.results_history: List[ProcessingResult] = []
    
    def process_species_folder(self, species_folder: Path, species_name: str = None,
                             max_images_per_species: int = 20) -> ProcessingResult:
        """
        Process folder berisi gambar untuk satu spesies
        
        Args:
            species_folder: Path ke folder spesies
            species_name: Nama spesies (jika None, akan menggunakan nama folder)
            max_images_per_species: Maksimal gambar per spesies
        """
        if species_name is None:
            species_name = species_folder.name
        
        logger.info(f"🔄 Processing species: {species_name} from {species_folder}")
        
        start_time = datetime.now()
        
        # Find images
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(species_folder.glob(f"*{ext}"))
        
        if not image_files:
            logger.warning(f"⚠️ No images found in {species_folder}")
            return ProcessingResult(
                species_name=species_name,
                total_images=0,
                successful=0,
                failed=0,
                error_messages=["No images found in folder"],
                processing_time=0.0
            )
        
        # Limit images
        total_images = min(len(image_files), max_images_per_species)
        selected_images = image_files[:total_images]
        
        logger.info(f"📁 Found {len(image_files)} images, processing {total_images}")
        
        # Process images
        successful = 0
        failed = 0
        error_messages = []
        
        for i, image_path in enumerate(selected_images):
            logger.info(f"   Processing {i+1}/{total_images}: {image_path.name}")
            
            try:
                success = self.system.add_fish_to_database_permanent(
                    str(image_path), 
                    species_name, 
                    annotation_source="batch_processing"
                )
                
                if success:
                    successful += 1
                    logger.info(f"     ✅ Success")
                else:
                    failed += 1
                    error_messages.append(f"Failed to process {image_path.name}")
                    logger.warning(f"     ❌ Failed")
                    
            except Exception as e:
                failed += 1
                error_msg = f"Exception processing {image_path.name}: {str(e)}"
                error_messages.append(error_msg)
                logger.error(f"     ❌ Error: {e}")
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        result = ProcessingResult(
            species_name=species_name,
            total_images=total_images,
            successful=successful,
            failed=failed,
            error_messages=error_messages,
            processing_time=processing_time
        )
        
        self.results_history.append(result)
        
        logger.info(f"✅ Species '{species_name}' completed: {successful}/{total_images} successful")
        return result
    
    def process_dataset_folder(self, dataset_folder: Path, 
                             species_mapping: Dict[str, str] = None,
                             max_images_per_species: int = 20,
                             max_species: int = 50) -> List[ProcessingResult]:
        """
        Process entire dataset folder dengan multiple spesies
        
        Args:
            dataset_folder: Path ke folder dataset
            species_mapping: Mapping dari nama folder ke nama spesies yang benar
            max_images_per_species: Maksimal gambar per spesies
            max_species: Maksimal jumlah spesies yang diproses
        """
        logger.info(f"🔄 Processing dataset folder: {dataset_folder}")
        
        if not dataset_folder.exists():
            logger.error(f"❌ Dataset folder not found: {dataset_folder}")
            return []
        
        # Find species folders
        species_folders = [d for d in dataset_folder.iterdir() if d.is_dir()]
        
        if not species_folders:
            logger.warning(f"⚠️ No species folders found in {dataset_folder}")
            return []
        
        # Limit species
        total_species = min(len(species_folders), max_species)
        selected_species = species_folders[:total_species]
        
        logger.info(f"🐟 Found {len(species_folders)} species folders, processing {total_species}")
        
        # Create backup before processing
        logger.info("💾 Creating backup before batch processing...")
        backup_path = self.db_utils.create_backup("before_batch_processing")
        if backup_path:
            logger.info(f"✅ Backup created: {backup_path}")
        
        # Process each species
        results = []
        
        for i, species_folder in enumerate(selected_species):
            logger.info(f"\n--- Processing Species {i+1}/{total_species} ---")
            
            # Get species name
            species_name = species_folder.name
            if species_mapping and species_name in species_mapping:
                species_name = species_mapping[species_name]
            
            # Process species
            try:
                result = self.process_species_folder(
                    species_folder, 
                    species_name, 
                    max_images_per_species
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"❌ Error processing species {species_name}: {e}")
                error_result = ProcessingResult(
                    species_name=species_name,
                    total_images=0,
                    successful=0,
                    failed=1,
                    error_messages=[f"Exception during processing: {str(e)}"],
                    processing_time=0.0
                )
                results.append(error_result)
        
        logger.info(f"\n🎯 Dataset processing completed: {len(results)} species processed")
        return results
    
    def generate_processing_report(self, results: List[ProcessingResult] = None) -> Dict[str, Any]:
        """Generate comprehensive processing report"""
        if results is None:
            results = self.results_history
        
        if not results:
            return {"error": "No processing results available"}
        
        # Calculate summary statistics
        total_species = len(results)
        total_images = sum(r.total_images for r in results)
        total_successful = sum(r.successful for r in results)
        total_failed = sum(r.failed for r in results)
        total_time = sum(r.processing_time for r in results)
        
        # Species with best/worst success rates
        species_by_success = sorted(results, key=lambda x: x.success_rate, reverse=True)
        best_species = species_by_success[0] if species_by_success else None
        worst_species = species_by_success[-1] if species_by_success else None
        
        # Error analysis
        all_errors = []
        for result in results:
            all_errors.extend(result.error_messages)
        
        # Generate report
        report = {
            "processing_summary": {
                "total_species_processed": total_species,
                "total_images_processed": total_images,
                "total_successful": total_successful,
                "total_failed": total_failed,
                "overall_success_rate": (total_successful / total_images * 100) if total_images > 0 else 0,
                "total_processing_time_seconds": total_time,
                "average_time_per_image": total_time / total_images if total_images > 0 else 0
            },
            "species_results": [
                {
                    "species_name": r.species_name,
                    "total_images": r.total_images,
                    "successful": r.successful,
                    "failed": r.failed,
                    "success_rate": r.success_rate,
                    "processing_time": r.processing_time,
                    "errors": len(r.error_messages)
                }
                for r in results
            ],
            "performance_analysis": {
                "best_performing_species": {
                    "name": best_species.species_name if best_species else None,
                    "success_rate": best_species.success_rate if best_species else None
                },
                "worst_performing_species": {
                    "name": worst_species.species_name if worst_species else None,
                    "success_rate": worst_species.success_rate if worst_species else None
                },
                "species_with_failures": len([r for r in results if r.failed > 0]),
                "species_100_percent_success": len([r for r in results if r.success_rate == 100.0])
            },
            "error_analysis": {
                "total_errors": len(all_errors),
                "unique_error_types": len(set(all_errors)),
                "most_common_errors": self._count_common_errors(all_errors)
            },
            "database_status": self.system.get_database_stats(),
            "report_generated": datetime.now().isoformat()
        }
        
        return report
    
    def _count_common_errors(self, errors: List[str], top_n: int = 5) -> List[Dict[str, Any]]:
        """Count most common error messages"""
        from collections import Counter
        
        error_counts = Counter(errors)
        common_errors = error_counts.most_common(top_n)
        
        return [
            {"error_message": error, "count": count}
            for error, count in common_errors
        ]
    
    def save_processing_report(self, report: Dict[str, Any], output_path: str) -> bool:
        """Save processing report to JSON file"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Processing report saved to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving report: {e}")
            return False

def main():
    """Demo batch processing"""
    print("🔄 Batch Processing System Demo")
    print("=" * 60)
    
    # Initialize processor
    processor = BatchProcessor()
    
    # Show initial database stats
    print("\n📊 Initial Database Statistics:")
    initial_stats = processor.system.get_database_stats()
    print(f"   Total embeddings: {initial_stats['total_embeddings']}")
    print(f"   Total species: {initial_stats['total_labels']}")
    
    # Demo 1: Process single species folder
    print("\n" + "="*60)
    print("🐟 DEMO 1: Processing Single Species Folder")
    print("="*60)
    
    # Example: process images folder as a test species
    test_folder = Path("../images")
    if test_folder.exists():
        result = processor.process_species_folder(
            test_folder, 
            "Test Species", 
            max_images_per_species=3
        )
        
        print(f"\n📋 Processing Result:")
        print(f"   Species: {result.species_name}")
        print(f"   Images processed: {result.total_images}")
        print(f"   Successful: {result.successful}")
        print(f"   Failed: {result.failed}")
        print(f"   Success rate: {result.success_rate:.1f}%")
        print(f"   Processing time: {result.processing_time:.2f}s")
        
        if result.error_messages:
            print(f"   Errors:")
            for error in result.error_messages[:3]:  # Show first 3 errors
                print(f"     - {error}")
    else:
        print(f"⚠️ Test folder not found: {test_folder}")
    
    # Demo 2: Process dataset folder (with limit for demo)
    print("\n" + "="*60)
    print("🗂️ DEMO 2: Processing Dataset Folder")
    print("="*60)
    
    dataset_folder = Path("../dataset/ikan_db_v1/images")
    if dataset_folder.exists():
        # Create species mapping for Indonesian names
        species_mapping = {
            "bandeng": "Bandeng",
            "mujair": "Mujair",
            "nila": "Nila",
            "lele": "Lele",
            "mas": "Ikan Mas"
        }
        
        results = processor.process_dataset_folder(
            dataset_folder,
            species_mapping=species_mapping,
            max_images_per_species=2,  # Limit for demo
            max_species=3  # Limit for demo
        )
        
        print(f"\n📋 Batch Processing Results:")
        for result in results:
            print(f"   {result.species_name}: {result.successful}/{result.total_images} ({result.success_rate:.1f}%)")
    else:
        print(f"⚠️ Dataset folder not found: {dataset_folder}")
    
    # Demo 3: Generate and save report
    print("\n" + "="*60)
    print("📊 DEMO 3: Processing Report")
    print("="*60)
    
    if processor.results_history:
        report = processor.generate_processing_report()
        
        print(f"\n📋 Processing Summary:")
        summary = report['processing_summary']
        print(f"   Total species: {summary['total_species_processed']}")
        print(f"   Total images: {summary['total_images_processed']}")
        print(f"   Success rate: {summary['overall_success_rate']:.1f}%")
        print(f"   Processing time: {summary['total_processing_time_seconds']:.2f}s")
        
        # Save report
        report_path = "batch_processing_report.json"
        success = processor.save_processing_report(report, report_path)
        if success:
            print(f"   📄 Report saved: {report_path}")
    else:
        print("⚠️ No processing results available for report")
    
    # Show final database stats
    print("\n📊 Final Database Statistics:")
    final_stats = processor.system.get_database_stats()
    print(f"   Total embeddings: {final_stats['total_embeddings']}")
    print(f"   Total species: {final_stats['total_labels']}")
    
    added_embeddings = final_stats['total_embeddings'] - initial_stats['total_embeddings']
    added_species = final_stats['total_labels'] - initial_stats['total_labels']
    
    print(f"\n🎯 Changes:")
    print(f"   Added embeddings: +{added_embeddings}")
    print(f"   Added species: +{added_species}")
    
    print("\n✅ Batch Processing Demo Completed!")

if __name__ == "__main__":
    main()