import requests
import cv2
import numpy as np
import base64
import sys

def test_pipeline():
    # 1. Create a dummy shelf image
    img = np.ones((800, 1200, 3), dtype=np.uint8) * 200 # light gray background
    
    # Draw some "products"
    for i in range(5):
        x1, y1 = 100 + i*200, 300
        x2, y2 = 250 + i*200, 500
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), -1) # Blue boxes
        cv2.putText(img, "Pepsi", (x1+10, y1+50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
    img_path = "test_shelf.jpg"
    cv2.imwrite(img_path, img)
    print(f"Created dummy image {img_path}")
    
    # 2. Send to /detect
    url = "http://127.0.0.1:8000/detect"
    try:
        with open(img_path, "rb") as f:
            files = {"file": ("test_shelf.jpg", f, "image/jpeg")}
            data = {"confidence": "0.1", "ocr_enabled": "true"}
            print(f"Sending request to {url}...")
            response = requests.post(url, files=files, data=data)
            
        if response.status_code == 200:
            print("Request successful!")
            result = response.json()
            
            print(f"Total Products Detected: {result.get('total_products')}")
            print(f"Anomalies: {len(result.get('anomalies', []))}")
            for a in result.get('anomalies', []):
                print(f" - {a.get('type')}: {a.get('description')}")
            
            print("Inventory:")
            inv = result.get('product_inventory', {})
            print(f" - Identified: {inv.get('total_identified')}")
            print(f" - Unidentified: {inv.get('total_unidentified')}")
            
            if "image_b64" in result:
                print("Image returned successfully (base64).")
            else:
                print("WARNING: image_b64 missing from response!")
                sys.exit(1)
        else:
            print(f"Error {response.status_code}: {response.text}")
            sys.exit(1)
    except Exception as e:
        print(f"Exception during request: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_pipeline()
