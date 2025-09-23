#!/usr/bin/env python3
"""
Debug Classification Issue
Analyze why Mujair classification is not working
"""

import sys
sys.path.append('..')
import torch
import numpy as np
from advanced_fish_recognition import AdvancedFishSystem

def debug_classification():
    print("🔍 Debug Classification Issue")
    print("=" * 40)
    
    # Initialize system
    system = AdvancedFishSystem()
    
    if not system.initialized:
        print("❌ System initialization failed!")
        return
    
    # Load database directly to analyze
    print("\n📊 Analyzing database structure...")
    db_data = torch.load('../models/classification/database.pt')
    embeddings = db_data[0]  # All embeddings
    internal_ids = db_data[1]
    image_ids = db_data[2]  
    annotation_ids = db_data[3]
    drawn_fish_ids = db_data[4]
    keys = db_data[5]
    
    print(f"Total embeddings: {embeddings.shape[0]}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    
    # Check for Mujair embeddings (should be the last 3)
    print(f"\n🐟 Looking for Mujair embeddings...")
    print(f"Total length: {len(internal_ids)}")
    
    # Check last few entries
    print(f"\nLast 5 internal IDs: {internal_ids[-5:]}")
    
    # Look for Mujair in keys
    mujair_entries = []
    for key, value in keys.items():
        if isinstance(value, dict) and 'label' in value:
            if 'mujair' in value['label'].lower():
                mujair_entries.append((key, value))
    
    print(f"\n🔍 Found {len(mujair_entries)} Mujair entries in keys:")
    for key, value in mujair_entries:
        print(f"  Key {key}: {value}")
    
    # Get the last 3 embeddings (should be Mujair)
    if len(embeddings) >= 3:
        last_3_embeddings = embeddings[-3:]
        print(f"\n📊 Last 3 embeddings (should be Mujair):")
        for i, emb in enumerate(last_3_embeddings):
            norm = torch.norm(emb).item()
            print(f"  Embedding {len(embeddings)-3+i}: norm = {norm:.4f}")
    
    # Test similarity calculation manually
    print(f"\n🧮 Manual similarity test...")
    
    # Get embedding for mujair1.jpg
    print(f"Generating embedding for mujair1.jpg...")
    
    import cv2
    img = cv2.imread('../images/mujair1.jpg')
    if img is None:
        print("❌ Cannot load test image")
        return
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Detect fish
    detection_results = system.detector.predict(img_rgb)
    if not detection_results or len(detection_results[0]) == 0:
        print("❌ No fish detected")
        return
    
    best_detection = detection_results[0][0]
    cropped_bgr = best_detection.get_mask_BGR()
    
    # Generate embedding using classifier model directly
    from PIL import Image
    cropped_pil = Image.fromarray(cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB))
    image_tensor = system.classifier.loader(cropped_pil).unsqueeze(0)
    
    with torch.no_grad():
        test_embedding, _ = system.classifier.model(image_tensor)
        test_embedding = test_embedding[0]  # Shape: [128]
    
    print(f"Test embedding norm: {torch.norm(test_embedding).item():.4f}")
    
    # Calculate similarity with all embeddings
    similarities = torch.cosine_similarity(test_embedding.unsqueeze(0), embeddings)
    
    # Get top 10 most similar
    top_similarities, top_indices = torch.topk(similarities, 10)
    
    print(f"\n🏆 Top 10 most similar embeddings:")
    for i, (sim, idx) in enumerate(zip(top_similarities, top_indices)):
        idx = idx.item()
        sim = sim.item()
        
        # Try to find what species this is
        species_info = "Unknown"
        if idx < len(internal_ids):
            internal_id = internal_ids[idx]
            # Look in keys for this ID or similar
            for key, value in keys.items():
                if isinstance(value, dict) and 'label' in value:
                    species_info = value['label']
                    break
        
        print(f"  {i+1}. Index {idx}: similarity = {sim:.6f}, species ≈ {species_info}")
    
    # Check if any of top matches are Mujair
    mujair_in_top = False
    for idx in top_indices[:5]:  # Check top 5
        if idx.item() >= len(embeddings) - 3:  # Last 3 should be Mujair
            mujair_in_top = True
            break
    
    if mujair_in_top:
        print("✅ Mujair embeddings found in top 5 similarities")
    else:
        print("❌ Mujair embeddings NOT in top 5 similarities")
        print("💡 This explains why classification is wrong")
        
        # Check actual similarity with Mujair embeddings
        print(f"\n🔍 Similarity with Mujair embeddings:")
        for i in range(3):
            mujair_idx = len(embeddings) - 3 + i
            mujair_similarity = similarities[mujair_idx].item()
            print(f"  Mujair embedding {i+1}: similarity = {mujair_similarity:.6f}")

if __name__ == "__main__":
    debug_classification()