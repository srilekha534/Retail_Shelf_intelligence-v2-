import numpy as np
import torch
import sys
from pathlib import Path

# Add root directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from src.anomaly.dl_detector import DLAnomalyDetector

def test():
    print("Initializing DL Anomaly Detector...")
    detector = DLAnomalyDetector(device="cpu", precision="fp32")
    
    print("Creating fake normal crops...")
    fake_crops = [np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8) for _ in range(20)]
    
    print("Training autoencoder (1 epoch)...")
    detector.train(fake_crops, epochs=1, lr=1e-3)
    
    print("Testing inference...")
    scores = detector.predict_batch(fake_crops[:5])
    print("Anomaly Scores:", scores)
    print("Success!")

if __name__ == "__main__":
    test()
