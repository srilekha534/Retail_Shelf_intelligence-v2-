import requests

url = "http://localhost:8000/detect"
files = {'file': ('test.jpg', open('data/history_images/05fce25fdf1e4279943a607afd24ed88_original.jpg', 'rb'), 'image/jpeg')}
data = {'confidence': '0.5', 'ocr_enabled': 'true'}

response = requests.post(url, files=files, data=data)

print(f"Status Code: {response.status_code}")
if response.status_code != 200:
    print(response.text)
else:
    print("Success!")
    data = response.json()
    inventory = data.get("product_inventory", {})
    counts = inventory.get("counts_by_name", {})
    print(f"\nTotal Identified: {inventory.get('total_identified', 0)}")
    print(f"Total Unidentified: {inventory.get('total_unidentified', 0)}")
    print("Product Counts:")
    for name, count in counts.items():
        print(f"  - {name}: {count}")
    
    print("\nRaw Products:")
    for p in inventory.get("products", [])[:5]:
        print(f"  - {p['name']} (OCR: {p['ocr_confidence']}, texts: {p.get('all_texts', [])})")
