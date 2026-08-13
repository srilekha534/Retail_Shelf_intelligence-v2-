import numpy as np
import cv2
import sys
from pathlib import Path

# Add root directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from api.main import _run_detection

def test():
    print("Creating fake normal crop...")
    img = np.ones((800, 1200, 3), dtype=np.uint8) * 200 # light gray background
    for i in range(5):
        x1, y1 = 100 + i*200, 300
        x2, y2 = 250 + i*200, 500
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), -1)
        cv2.putText(img, "Pepsi", (x1+10, y1+50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
    print("Running detection...")
    try:
        res = _run_detection(img, "test_image.jpg", conf=0.1, ocr_enabled=False)
        print("Success! Keys in response:", res.keys())
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test()
