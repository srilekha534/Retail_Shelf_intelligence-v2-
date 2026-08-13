import requests
import os

url = 'http://localhost:8000/detect'
image_path = r'd:\retail_shelf_intelligence\data\history_images\00e5cc229ec14f6199ce8101e9f18b71_original.jpg' 

print(f'Sending {image_path} to API...')
with open(image_path, 'rb') as f:
    files = {'file': ('image.jpg', f, 'image/jpeg')}
    data = {'confidence': 0.25, 'ocr_enabled': 'true', 'detect_anomalies': 'true'}
    try:
        response = requests.post(url, files=files, data=data, timeout=60)
        print('Status code:', response.status_code)
        if response.status_code == 200:
            result = response.json()
            print('Detected products:', result.get('total_products'))
            print('Total Identified:', result.get('product_inventory', {}).get('total_identified'))
            print('Anomalies:', len(result.get('anomalies', [])))
        else:
            print('Error:', response.text)
    except Exception as e:
        print('Failed to request:', e)
