"""
Universal Dental AI - Complete Offline Model Training Engine
Initializes a blank YOLOv11-Seg architecture fully offline to bypass 
server proxy/network drop restrictions and launches training instantly.
"""

import os
from ultralytics import YOLO

def train_dental_model_offline():
    print("=" * 60)
    print("UNIVERSAL DENTAL AI - OFFLINE MODEL TRAINING INITIALIZATION")
    print("=" * 60)

    # Step 1: Initialize a completely blank YOLOv11-Seg architecture using configuration specs
    # This enforces 100% offline compilation and skips trying to download remote weights over proxy
    print("\n[1/2] Compiling offline neural network architecture layers...")
    model = YOLO("yolo11n-seg.yaml") 

    dataset_yaml_path = os.path.join(os.path.dirname(__file__), "dataset.yaml")
    if not os.path.exists(dataset_yaml_path):
        print(f" -> ERROR: Configuration file missing at {dataset_yaml_path}")
        return

    # Step 2: Launch transfer learning epochs directly over your extracted local dataset matrix
    print("\n[2/2] Launching offline deep learning optimization pipeline...")
    model.train(
        data=dataset_yaml_path,
        epochs=100,
        imgsz=1024,
        batch=4,
        device="cpu", # Leverages active CPU core threads for absolute standalone environment stability
        workers=2,
        name="universal_dental_ai_v1"
    )

    print("\n SUCCESS: Model training pipeline finished.")
    print("=" * 60)

if __name__ == "__main__":
    train_dental_model_offline()
