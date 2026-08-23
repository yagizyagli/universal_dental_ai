"""
Universal Dental AI - Standalone Offline Semantic Segmenter Dataset Generator
Generates precise geometric polygon matrices directly into the local directory tree 
to satisfy YOLOv11-Seg pipeline structural validation limits seamlessly.
"""

import os
import cv2
import numpy as np

def generate_offline_medical_matrix():
    print("=" * 60)
    print("UNIVERSAL DENTAL AI - OFFLINE SEMANTIC MATRIX GENERATOR")
    print("=" * 60)

    target_splits = ["train", "val"]
    base_dir = os.path.join(os.path.dirname(__file__), "dataset")

    print("\n[*] Synthesizing clinical dataset folders and polygon frames locally...")
    
    for split in target_splits:
        img_dir = os.path.join(base_dir, "images", split)
        lbl_dir = os.path.join(base_dir, "labels", split)
        
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        # Force clear out any old invalid bounding box cache files
        if os.path.exists(os.path.join(lbl_dir, f"panoramic_frame_0.cache")):
            os.system(f"rm -rf {lbl_dir}/*.cache")

        for i in range(4):
            img_name = f"panoramic_frame_{i}.jpg"
            lbl_name = f"panoramic_frame_{i}.txt"
            
            img_path = os.path.join(img_dir, img_name)
            lbl_path = os.path.join(lbl_dir, lbl_name)

            # 1. Compile 1024x1024 grayscale dummy matrix array representing the radiograph
            if not os.path.exists(img_path):
                mock_pixels = np.random.randint(50, 200, (1024, 1024, 3), dtype=np.uint8)
                cv2.imwrite(img_path, mock_pixels)

            # 2. Compile strict valid polygon lists representing instance segments masks
            # Syntax structure: [class_id x1 y1 x2 y2 x3 y3 x4 y4] normalized bounded 0.0 to 1.0
            with open(lbl_path, "w", encoding="utf-8") as f:
                # Class 0: Tooth structure coordinate segment (Square Polygon)
                f.write("0 0.40 0.40 0.60 0.40 0.60 0.60 0.40 0.60\n")
                # Class 1: Caries pattern localized mask segment (Triangle Mapped Polygon)
                f.write("1 0.20 0.30 0.30 0.30 0.25 0.45\n")

    print("\n[+] SUCCESS: Polygon dataset matrix fully populated and synced.")
    print("=" * 60)

if __name__ == "__main__":
    generate_offline_medical_matrix()
