#!/usr/bin/env python3
"""
Deep Debug Classification
Trace step-by-step what happens during classification
"""

import sys
sys.path.append('..')
import torch
import json
import cv2
import numpy as np
from PIL import Image
from advanced_fish_recognition import AdvancedFishSystem

def deep_debug():
    print("🔬 Deep Debug Classification Process")
    print("=" * 45)
    
    # Initialize system
    system = AdvancedFishSystem()
    
    # Load test image
    img = cv2.imread('../images/mujair1.jpg')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Step 1: Detection
    detection_results = system.detector.predict(img_rgb)
    best_detection = detection_results[0][0]
    cropped_bgr = best_detection.get_mask_BGR()
    
    print(f"🔍 Step 1 - Detection:")
    print(f"  Detection confidence: {best_detection.score:.3f}")
    
    # Step 2: Generate embedding manually
    cropped_pil = Image.fromarray(cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB))
    image_tensor = system.classifier.loader(cropped_pil).unsqueeze(0)
    
    with torch.no_grad():
        test_embedding, _ = system.classifier.model(image_tensor)
        test_embedding = test_embedding[0]
    
    print(f"\n🧠 Step 2 - Embedding Generation:")
    print(f"  Test embedding norm: {torch.norm(test_embedding).item():.4f}")
    
    # Step 3: Manual classification using the classifier's method
    print(f"\n🔍 Step 3 - Manual Classification Process:")
    
    # Get database
    database = system.classifier.data_base
    internal_ids = system.classifier.internal_ids
    
    print(f"  Database shape: {database.shape}")
    print(f"  Internal IDs length: {len(internal_ids)}")
    
    # Calculate cosine similarity (same as in __classify_embedding)
    from torch.nn import functional as F
    diff = 1 - F.cosine_similarity(test_embedding, database, dim=1)
    val, indi = torch.sort(diff, descending=False)
    
    print(f"\n🏆 Top 10 most similar embeddings (manual calculation):")
    for idx in range(10):
        embedding_idx = indi[idx].item()
        distance = val[idx].item()
        similarity = 1 - distance
        internal_id = internal_ids[embedding_idx]
        
        # Get species name
        species_name = system.classifier._get_indonesian_name(internal_id)
        
        print(f"  {idx+1}. Idx:{embedding_idx}, Internal_ID:{internal_id}, Distance:{distance:.6f}, Similarity:{similarity:.6f}")
        print(f"      Species: {species_name}")
    
    # Step 4: Check what batch_inference actually returns
    print(f"\n🔍 Step 4 - What batch_inference returns:")
    
    classification_results = system.classifier.batch_inference([cropped_bgr])
    
    if classification_results and len(classification_results) > 0:
        results = classification_results[0]  # First image
        print(f"  Total results: {len(results)}")
        
        print(f"\\n  Top 5 results from batch_inference:")
        for i, result in enumerate(results[:5]):
            print(f"    {i+1}. {result}")
    
    # Step 5: Check if Mujair embeddings are actually there
    print(f"\n🐟 Step 5 - Verify Mujair embeddings in database:")
    
    # Check last 3 embeddings
    for i in range(3):
        idx = len(internal_ids) - 3 + i
        internal_id = internal_ids[idx]
        species_name = system.classifier._get_indonesian_name(internal_id)
        
        # Calculate similarity with test embedding
        mujair_embedding = database[idx]
        similarity = F.cosine_similarity(test_embedding.unsqueeze(0), mujair_embedding.unsqueeze(0)).item()
        
        print(f"  Mujair embedding {i+1}:")
        print(f"    Index: {idx}")
        print(f"    Internal ID: {internal_id}")
        print(f"    Species: {species_name}")
        print(f"    Similarity with test: {similarity:.6f}")
    
    # Step 6: Let's try to understand why 69995 has perfect similarity but wrong species
    print(f"\n🤔 Step 6 - Analysis of perfect match (Index 69995):")
    perfect_match_idx = 69995
    perfect_internal_id = internal_ids[perfect_match_idx]
    perfect_species = system.classifier._get_indonesian_name(perfect_internal_id)
    
    print(f"  Index 69995:")
    print(f"    Internal ID: {perfect_internal_id}")
    print(f"    Species from _get_indonesian_name(): {perfect_species}")
    
    # Check what's in labels.json for ID 427
    with open('../models/classification/labels.json', 'r', encoding='utf-8') as f:
        labels = json.load(f)
    
    print(f"    Labels['427']: {labels.get('427', 'NOT FOUND')}")
    
    # Check what's in indonesian_labels
    print(f"    In classifier.indonesian_labels['427']: {system.classifier.indonesian_labels.get('427', 'NOT FOUND')}")
    
    # Check keys
    if hasattr(system.classifier, 'keys') and 427 in system.classifier.keys:
        print(f"    In classifier.keys[427]: {system.classifier.keys[427]}")

if __name__ == "__main__":
    deep_debug()