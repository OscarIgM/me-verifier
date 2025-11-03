# scripts/embeddings.py
import os
import numpy as np
import pandas as pd
from PIL import Image
from facenet_pytorch import InceptionResnetV1
import torch

def get_embeddings(cropped_dir):
    """
    Genera embeddings faciales para cada imagen recortada en data/cropped/
    Devuelve un array de embeddings y un DataFrame con etiquetas.
    """
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    
    embeddings = []
    labels = []

    for label in os.listdir(cropped_dir):
        folder = os.path.join(cropped_dir, label)
        if not os.path.isdir(folder):
            continue
        
        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            try:
                img = Image.open(img_path).convert('RGB')
            except:
                print(f"[SKIP] No se puede abrir {img_name}")
                continue

            try:
                img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
                img_tensor = img_tensor.unsqueeze(0)  
                with torch.no_grad():
                    embedding = resnet(img_tensor).numpy().flatten()
                embeddings.append(embedding)
                labels.append(label)
                print(f"[OK] {img_name} procesado")
            except Exception as e:
                print(f"[SKIP] Error procesando {img_name}: {e}")
                continue

    embeddings = np.array(embeddings)
    df = pd.DataFrame({'label': labels})
    return embeddings, df

if __name__ == "__main__":
    cropped_dir = 'data/cropped'
    embeddings, df = get_embeddings(cropped_dir)

    os.makedirs('models', exist_ok=True)

    np.save('models/embeddings.npy', embeddings)
    df.to_csv('models/labels.csv', index=False)

    print("✅ Embeddings guardados en models/embeddings.npy y labels.csv")
    print(f"Total embeddings generados: {len(embeddings)}")
