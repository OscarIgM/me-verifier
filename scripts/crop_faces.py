# scripts/crop_faces.py
import os
from PIL import Image
from facenet_pytorch import MTCNN
import argparse
import torch

def crop_faces(input_dir, output_dir):
    mtcnn = MTCNN(image_size=160, margin=20)  
    os.makedirs(output_dir, exist_ok=True)

    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        try:
            img = Image.open(img_path).convert('RGB')
        except:
            print(f"[SKIP] No se puede abrir {img_name}")
            continue

        face_tensor = mtcnn(img)  
        if face_tensor is not None:
            face_img = Image.fromarray((face_tensor.permute(1, 2, 0).numpy() * 255).astype('uint8'))
            face_img.save(os.path.join(output_dir, img_name))
            print(f"[OK] {img_name} recortado")
        else:
            print(f"[SKIP] No se detectó rostro en {img_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Carpeta de imágenes originales")
    parser.add_argument("--output", required=True, help="Carpeta donde guardar recortes")
    args = parser.parse_args()

    crop_faces(args.input, args.output)
