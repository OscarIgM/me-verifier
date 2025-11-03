from sklearn.datasets import fetch_openml
import numpy as np
from PIL import Image
import os


data = fetch_openml("olivetti_faces", version=1, as_frame=False)
X = data.data  # Shape (400, 4096)

images = X.reshape(-1, 64, 64)

# Crear carpeta para guardar las imágenes
output_dir = "olivetti_faces_sklearn"
os.makedirs(output_dir, exist_ok=True)

# Guardar cada imagen
for i, img_array in enumerate(images):
    img = (img_array * 255).astype("uint8")
    img_pil = Image.fromarray(img)
    img_pil.save(os.path.join(output_dir, f"face_{i+1}.png"))

print("¡Las 400 imágenes se han guardado correctamente!")
