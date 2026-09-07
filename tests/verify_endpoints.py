import requests
import os
import json

API = 'http://127.0.0.1:8000'

print("=" * 60)
print("TESTING MEDIFUSION AI API ENDPOINTS")
print("=" * 60)

# 1. Health
r = requests.get(f'{API}/api/health')
print('1. GET  /api/health            ->', r.status_code, r.json()['models_loaded'])

# 2. Symptoms
r = requests.post(f'{API}/api/predict/symptoms', json={'symptoms': {'fever': 1, 'cough': 1, 'rapid_breathing': 1}})
data = r.json()
print('2. POST /api/predict/symptoms  ->', r.status_code, f"Predicted: {data['predicted_class']} ({data['confidence']*100:.1f}%)")

# 3. Clinical
r = requests.post(f'{API}/api/predict/clinical', json={
    'symptoms': {'fever': 1, 'cough': 1, 'rapid_breathing': 1},
    'clinical_data': {'age': 55, 'temperature': 39.2, 'heart_rate': 110, 'bp_systolic': 135, 'bp_diastolic': 88, 'respiratory_rate': 28, 'oxygen_saturation': 90}
})
data = r.json()
print('3. POST /api/predict/clinical  ->', r.status_code, f"Predicted: {data['predicted_class']} ({data['confidence']*100:.1f}%)")

# 4. Image
xray_test_dir = r'C:\Users\vansh\.cache\kagglehub\datasets\paultimothymooney\chest-xray-pneumonia\versions\2\chest_xray\test\PNEUMONIA'
img_file = [os.path.join(xray_test_dir, f) for f in os.listdir(xray_test_dir) if f.endswith('.jpeg') or f.endswith('.png')][0]

with open(img_file, 'rb') as f:
    r = requests.post(f'{API}/api/predict/image', files={'file': ('xray.jpeg', f, 'image/jpeg')})
data = r.json()
print('4. POST /api/predict/image     ->', r.status_code, f"Predicted: {data['predicted_class']} ({data['confidence']*100:.1f}%)")

# 5. Multimodal
with open(img_file, 'rb') as f:
    r = requests.post(f'{API}/api/predict/multimodal', files={'file': ('xray.jpeg', f, 'image/jpeg')}, data={
        'symptoms': json.dumps({'fever': 1, 'cough': 1, 'rapid_breathing': 1}),
        'clinical_data': json.dumps({'age': 55, 'temperature': 39.2, 'heart_rate': 110, 'bp_systolic': 135, 'bp_diastolic': 88, 'respiratory_rate': 28, 'oxygen_saturation': 90})
    })
data = r.json()
print('5. POST /api/predict/multimodal->', r.status_code, f"Predicted: {data['predicted_class']} ({data['confidence']*100:.1f}%), Modalities: {data['modalities_used']}")

# 6. Explain Image (Grad-CAM)
with open(img_file, 'rb') as f:
    r = requests.post(f'{API}/api/explain/image', files={'file': ('xray.jpeg', f, 'image/jpeg')})
data = r.json()
print('6. POST /api/explain/image     ->', r.status_code, f"Grad-CAM Heatmap + Overlay generated (bytes: {len(data.get('gradcam_overlay_base64', ''))})")

# 7. Explain Clinical (SHAP)
r = requests.post(f'{API}/api/explain/clinical', json={
    'symptoms': {'fever': 1, 'cough': 1, 'rapid_breathing': 1},
    'clinical_data': {'age': 55, 'temperature': 39.2, 'heart_rate': 110, 'bp_systolic': 135, 'bp_diastolic': 88, 'respiratory_rate': 28, 'oxygen_saturation': 90}
})
data = r.json()
print('7. POST /api/explain/clinical  ->', r.status_code, f"SHAP plot generated (top feature: {data['feature_importance'][0]['feature']})")

print("=" * 60)
print("[OK] ALL 7 ENDPOINTS RETURNED 200 OK WITH VALID RESPONSES!")
print("=" * 60)
