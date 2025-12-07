# api/app.py
import os
import io
import time
from flask import Flask, request, jsonify
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import joblib
import numpy as np
from dotenv import load_dotenv

load_dotenv()
MODEL_PATH = os.getenv('MODEL_PATH', 'models/model.joblib')
SCALER_PATH = os.getenv('SCALER_PATH', 'models/scaler.joblib')
THRESHOLD = float(os.getenv('THRESHOLD', 0.75))
MAX_MB = float(os.getenv('MAX_MB', 5))

app = Flask(__name__)

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
mtcnn = MTCNN(image_size=160, margin=20)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

@app.route('/healthz', methods=['GET'])
def healthz():
    return jsonify({"status": "ok"}), 200

@app.route('/verify', methods=['POST'])
def verify():
    start_time = time.time()

    if 'image' not in request.files:
        return jsonify({"error": "Archivo 'image' requerido"}), 400

    file = request.files['image']
    if file.content_type not in ['image/jpeg', 'image/png']:
        return jsonify({"error": "solo image/jpeg o image/png"}), 400

    if len(file.read()) > MAX_MB * 1024 * 1024:
        return jsonify({"error": f"Archivo excede {MAX_MB} MB"}), 400
    file.seek(0)

    try:
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
    except:
        return jsonify({"error": "No se pudo abrir la imagen"}), 400

    face = mtcnn(img)
    if face is None:
        return jsonify({"error": "No se detectó rostro"}), 400

    face_tensor = face.unsqueeze(0)  
    with torch.no_grad():
        embedding = resnet(face_tensor).numpy()
    embedding_scaled = scaler.transform(embedding)

    score = float(model.predict_proba(embedding_scaled)[0,1])
    is_me = score >= THRESHOLD
    if score < THRESHOLD:
        is_me = False  

    timing_ms = (time.time() - start_time) * 1000

    response = {
        "model_version": "me-verifier-v1",
        "is_me": is_me,
        "score": score,
        "threshold": THRESHOLD,
        "timing_ms": timing_ms
    }

    return jsonify(response), 200

if __name__ == "__main__":
    port = int(os.getenv('PORT', 8002))
    app.run(host='0.0.0.0', port=port, debug=True)
