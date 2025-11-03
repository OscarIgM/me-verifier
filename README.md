# Mi Verificador de Identidad por Imagen

Proyecto práctico 2 – Reconocimiento facial personal y verificación “¿soy yo?”

Este proyecto entrena un **modelo binario “yo vs no-yo”** usando embeddings faciales preentrenados (`facenet-pytorch`) y lo expone mediante un **endpoint REST Flask**.

---

## 📦 Tecnologías utilizadas

* Python 3.11+
* PyTorch
* facenet-pytorch (MTCNN + InceptionResnetV1)
* scikit-learn (Logistic Regression)
* Flask
* Pillow, NumPy, Pandas, Joblib
* Matplotlib (curvas ROC/PR)
* AWS EC2 (despliegue opcional)

---

## 🗂 Estructura del repositorio

```
me-verifier/
├─ api/
│  └─ app.py              # Flask API (/healthz, /verify)
├─ models/
│  ├─ model.joblib        # Clasificador entrenado
│  └─ scaler.joblib       # Escaler para embeddings
├─ data/
│  ├─ me/                 # Fotos propias crudas
│  ├─ not_me/             # Fotos negativas
│  └─ cropped/            # Rostros recortados
├─ reports/
│  ├─ metrics.json        # Métricas del entrenamiento
│  ├─ evaluation.json     # Evaluación completa
│  ├─ roc_curve.png
│  └─ pr_curve.png
├─ scripts/
│  ├─ crop_faces.py       # Recorte de rostros
│  ├─ embeddings.py       # Generación de embeddings
│  ├─ train.py            # Entrenamiento
│  └─ evaluate.py         # Evaluación y curvas
├─ README.md
└─ requirements.txt
```

---

## ⚡ Instalación

1. Clonar repositorio:

```bash
git clone <tu-repo-url>
cd me-verifier
```

2. Crear y activar entorno virtual:

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

3. Instalar dependencias:

```bash
pip install -r requirements.txt
```

---

## 🖼 Uso

### 1️⃣ Preparar datos

* Colocar fotos propias en `data/me/` y fotos negativas en `data/not_me/`.
* Recortar rostros:

```bash
python scripts/crop_faces.py --input data/me --output data/cropped/me
python scripts/crop_faces.py --input data/not_me --output data/cropped/not_me
```

* Generar embeddings:

```bash
python scripts/embeddings.py
```

---

### 2️⃣ Entrenamiento y evaluación

```bash
python scripts/train.py
python scripts/evaluate.py
```

* Genera:

  * `models/model.joblib`
  * `models/scaler.joblib`
  * `reports/metrics.json`
  * `reports/evaluation.json`
  * Curvas ROC/PR en `reports/`.

---

### 3️⃣ Ejecutar API Flask

```bash
python api/app.py
```

* Acceso en `http://127.0.0.1:5000/`

Endpoints:

| Endpoint   | Método | Descripción                                                     |
| ---------- | ------ | --------------------------------------------------------------- |
| `/healthz` | GET    | Verifica que la API esté funcionando                            |
| `/verify`  | POST   | Recibe imagen (form-data `image`) y devuelve JSON con resultado |

Ejemplo JSON de respuesta:

```json
{
  "model_version": "me-verifier-v1",
  "is_me": true,
  "score": 0.93,
  "threshold": 0.75,
  "timing_ms": 28.7
}
```

---

### 4️⃣ Prueba con Postman o curl

```bash
curl -F "image=@samples/selfie.jpg" http://127.0.0.1:5000/verify
```

---

## 📊 Resultados esperados

* Accuracy y AUC > 0.9
* Umbral óptimo `τ` calculado automáticamente
* Curvas ROC y PR en `reports/`
* Respuesta rápida (<50 ms en CPU para 1 rostro)

---

## ⚠️ Notas y mejoras

* Actualmente soporta **una sola cara por imagen**.
* Podría ampliarse a múltiples rostros por imagen.
* Considerar **enmascaramiento/privacidad** de fotos en producción.
* Para producción: usar **Gunicorn + Nginx** en AWS EC2.
