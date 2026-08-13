import numpy as np
import cv2
import sys
from pathlib import Path

# Add root directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from api.main import _run_detection
from src.detection.detector import Detection, DetectionResult

def mock_detect(self, img_array, conf):
    # Mock detector to return one product
    res = DetectionResult(image_width=img_array.shape[1], image_height=img_array.shape[0])
    res.detections.append(Detection([100.0, 100.0, 200.0, 200.0], 0, "product", 0.9))
    return res

def test():
    import src.detection.detector
    # Patch the real detector
    src.detection.detector.ShelfDetector.detect = mock_detect
    
    print("Creating fake normal crop...")
    img = np.ones((800, 1200, 3), dtype=np.uint8) * 200 # light gray background
        
    print("Running detection...")
    try:
        res = _run_detection(img, "test_image.jpg", conf=0.1, ocr_enabled=True)
        print("Success! Anomalies found:", len(res.get('anomalies', [])))
        print("Detections evaluated:", len(res.get('detections', [])))
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test()
