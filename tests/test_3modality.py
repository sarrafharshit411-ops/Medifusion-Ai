import requests
import json
import numpy as np
from PIL import Image
import io

# Create synthetic test radiograph
img = Image.new('RGB', (224, 224), color=(30, 40, 60))
img_bytes = io.BytesIO()
img.save(img_bytes, format='PNG')
img_bytes.seek(0)

files = {
    'file': ('test_xray.png', img_bytes, 'image/png'),
    'symptoms': (None, json.dumps({'fever': 1, 'cough': 1, 'rapid_breathing': 1, 'fatigue': 1})),
    'clinical_data': (None, json.dumps({
        'age': 55, 'temperature': 39.0, 'heart_rate': 104,
        'bp_systolic': 130, 'bp_diastolic': 85, 'respiratory_rate': 26,
        'oxygen_saturation': 91
    }))
}

res = requests.post('http://127.0.0.1:8000/api/predict/multimodal', files=files)
print('3-Modality Status code:', res.status_code)
data = res.json()
print('Modalities used:', data['modalities_used'])
print('Predicted class:', data['predicted_class'])
print('Confidence:', data['confidence'])
