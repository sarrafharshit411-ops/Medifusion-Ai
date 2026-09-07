import requests
import json

files = {
    'symptoms': (None, json.dumps({'fever': 1, 'cough': 1, 'rapid_breathing': 1, 'fatigue': 1})),
    'clinical_data': (None, json.dumps({
        'age': 55, 'temperature': 39.0, 'heart_rate': 104,
        'bp_systolic': 130, 'bp_diastolic': 85, 'respiratory_rate': 26,
        'oxygen_saturation': 91
    }))
}

res = requests.post('http://127.0.0.1:8000/api/predict/multimodal', files=files)
print('Status code:', res.status_code)
print('Response JSON:', res.json())
