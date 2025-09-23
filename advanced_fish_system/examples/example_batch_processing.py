#!/usr/bin/env python3
"""
Example: Batch Processing
Contoh batch processing untuk multiple species
"""

import sys
import os
from pathlib import Path
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from batch_processor import BatchProcessor
from advanced_fish_recognition import AdvancedFishSystem

def main():
    """Example: Batch process multiple fish species"""
    
    print("=== Batch Processing Example ===")
    print()
    
    # Initialize batch processor
    print("🔧 Initializing Batch Processor...")
    processor = BatchProcessor()
    
    if not processor.system.initialized:
        print("❌ Failed to initialize system")
        return
    
    print("✅ Batch processor initialized successfully!")
    print()
    
    # Get initial database stats
    print("📊 Initial Database Statistics:")
    stats = processor.system.get_database_stats()
    print(f"  - Total embeddings: {stats['total_embeddings']}")
    print(f"  - Total species: {stats['total_labels']}")
    print()
    
    # Example 1: Process single species folder
    print("🔄 Example 1: Processing Single Species Folder")
    print("-" * 50)
    
    # Adjust this path to your actual species folder
    species_folder = Path("../../images")  # Contains mujair images
    species_name = "Ikan Mujair"
    
    if species_folder.exists():
        print(f"Processing folder: {species_folder}")
        print(f"Species name: {species_name}")
        print()
        
        result = processor.process_species_folder(
            species_folder=species_folder,
            species_name=species_name,
            max_images_per_species=10  # Limit for example
        )
        
        print("📊 Processing Results:")
        print(f"  - Total images found: {result.total_images}")
        print(f"  - Successfully processed: {result.successful}")
        print(f"  - Failed: {result.failed}")
        print(f"  - Success rate: {result.success_rate:.1f}%")
        print(f"  - Processing time: {result.processing_time:.2f}s")
        print(f"  - Average time per image: {result.avg_time_per_image:.2f}s")
        
        if result.errors:
            print("  - Errors encountered:")
            for error in result.errors[:3]:  # Show first 3 errors
                print(f"    * {error}")
        print()
    
    else:
        print(f"❌ Species folder not found: {species_folder}")
        print("Please adjust the species_folder path")
        print()
    
    # Example 2: Process dataset with multiple species (if available)
    print("🔄 Example 2: Processing Multiple Species Dataset")
    print("-" * 50)
    
    # Check if dataset folder exists
    dataset_folder = Path("../../dataset/ikan_db_v1/images")
    
    if dataset_folder.exists() and any(dataset_folder.iterdir()):
        print(f"Processing dataset: {dataset_folder}")
        print()
        
        # Create species mapping for first few folders
        species_folders = [f for f in dataset_folder.iterdir() if f.is_dir()][:5]  # Limit to 5 for example
        
        species_mapping = {}
        for folder in species_folders:
            # Use folder name as species name (you can customize this)
            species_mapping[folder.name] = folder.name.replace('-', ' ').title()
        
        print("Species mapping:")
        for folder_name, species_name in species_mapping.items():
            print(f"  - {folder_name} -> {species_name}")
        print()
        
        results = processor.process_dataset_folder(
            dataset_folder=dataset_folder,
            species_mapping=species_mapping,
            max_images_per_species=5,  # Limit for example
            max_species=5
        )
        
        print("📊 Dataset Processing Results:")
        total_processed = sum(r.successful for r in results)
        total_images = sum(r.total_images for r in results)
        overall_success_rate = (total_processed / total_images * 100) if total_images > 0 else 0
        
        print(f"  - Total species processed: {len(results)}")
        print(f"  - Total images processed: {total_images}")
        print(f"  - Overall success rate: {overall_success_rate:.1f}%")
        print()
        
        # Show per-species results
        print("Per-species results:")
        for result in results:
            print(f"  - {result.species_name}: {result.successful}/{result.total_images} ({result.success_rate:.1f}%)")
        print()
        
        # Generate comprehensive report
        print("📋 Generating Processing Report...")
        report = processor.generate_processing_report(results)
        
        # Save report
        report_path = Path("processing_report.json")
        processor.save_processing_report(report, str(report_path))
        print(f"✅ Report saved to: {report_path}")
        print()
        
        # Display summary
        summary = report['processing_summary']
        print("📊 Processing Summary:")
        print(f"  - Overall success rate: {summary['overall_success_rate']:.1f}%")
        print(f"  - Total processing time: {summary['total_processing_time']:.2f}s")
        print(f"  - Average time per image: {summary['average_time_per_image']:.2f}s")
        print(f"  - Total species processed: {summary['total_species_processed']}")
        print()
        
        # Performance analysis
        performance = report['performance_analysis']
        if performance['best_performing_species']:
            print("🏆 Performance Analysis:")
            print(f"  - Best performing species: {performance['best_performing_species']['name']} ({performance['best_performing_species']['success_rate']:.1f}%)")
            if performance['species_100_percent_success']:
                print(f"  - Species with 100% success: {len(performance['species_100_percent_success'])}")
            print()
        
        # Error analysis
        errors = report['error_analysis']
        if errors['total_errors'] > 0:
            print("⚠️ Error Analysis:")
            print(f"  - Total errors: {errors['total_errors']}")
            if errors['most_common_errors']:
                print("  - Most common errors:")
                for error in errors['most_common_errors'][:3]:
                    print(f"    * {error['error_message']}: {error['count']} times")
            print()
    
    else:
        print(f"❌ Dataset folder not found or empty: {dataset_folder}")
        print("Skipping dataset processing example")
        print()
    
    # Get final database stats
    print("📊 Final Database Statistics:")
    final_stats = processor.system.get_database_stats()
    print(f"  - Total embeddings: {final_stats['total_embeddings']}")
    print(f"  - Total species: {final_stats['total_labels']}")
    print(f"  - New embeddings added: {final_stats['total_embeddings'] - stats['total_embeddings']}")
    print()
    
    print("✅ Batch processing example completed!")

if __name__ == "__main__":
    main()