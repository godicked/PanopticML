import io
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, Dataset


class Transformer(object):
    def __init__(self):
        import torch
        from transformers import logging
        logging.set_verbosity_error()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.processor = None
        self.model = None

    @property
    def can_handle_text(self):
        return False


class AutoTransformer(Transformer):
    def __init__(self, hugging_face_model=None):
        super().__init__()
        from transformers import AutoModel, AutoProcessor
        if hugging_face_model:
            self.model = AutoModel.from_pretrained(hugging_face_model).to(self.device)
            self.processor = AutoProcessor.from_pretrained(hugging_face_model)
            self.name = hugging_face_model

    @property
    def can_handle_text(self):
        return True

    def to_vector_batch(self, images) -> np.ndarray:
        """Version batch optimisée"""
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features.cpu().numpy()

    def to_text_vector(self, text: str) -> np.ndarray:
        inputs = self.processor(text=[text], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy().flatten()


class SIGLIPTransformer(AutoTransformer):
    def __init__(self):
        model_name = "google/siglip2-so400m-patch16-naflex"
        super().__init__(model_name)
        self.name = "SIGLIP"

    @property
    def can_handle_text(self):
        return True


class ImageDataset(Dataset):
    """Dataset personnalisé pour le chargement parallèle des images"""

    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            with open(img_path, 'rb') as f:
                image = Image.open(io.BytesIO(f.read())).convert('RGB')
            return image, str(img_path)
        except Exception as e:
            print(f"Erreur lors du chargement de {img_path}: {e}")
            # Retourner une image vide en cas d'erreur
            return Image.new('RGB', (224, 224), color='white'), str(img_path)


def collate_fn(batch):
    """Fonction pour grouper les images en batch"""
    images, paths = zip(*batch)
    return list(images), list(paths)


def get_images(folder):
    return [f for f in folder.iterdir() if
            f.suffix in ['.jpg', '.jpeg', '.png', '.gif'] and f.name != 'cropped_chat.png']


def generate_vectors_optimized(transformer: Transformer, images, batch_size=32, num_workers=4):
    """Version optimisée avec DataLoader"""

    # Créer le dataset et dataloader
    dataset = ImageDataset(images)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,  # Chargement parallèle des images
        pin_memory=True,  # Optimisation GPU
        collate_fn=collate_fn
    )

    vectors = []
    image_paths = []

    print(f"Traitement de {len(images)} images par batch de {batch_size}")
    print(f"Utilisation de {num_workers} workers pour le chargement")

    for batch_images, batch_paths in tqdm(dataloader, desc="Processing batches"):
        # Traitement par batch sur GPU
        batch_vectors = transformer.to_vector_batch(batch_images)

        vectors.extend(batch_vectors)
        image_paths.extend(batch_paths)

    return vectors, image_paths



if __name__ == "__main__":
    folder = r"D:\CorpusImage\documerica\extracted_images"
    # Configuration optimisée
    BATCH_SIZE = 16  # Ajustez selon votre VRAM
    NUM_WORKERS = 6  # Ajustez selon votre CPU

    print("Initialisation du modèle SIGLIP...")
    siglip = SIGLIPTransformer()

    print("Récupération des images...")
    images = get_images(Path(folder))
    print(f"Trouvé {len(images)} images")

    print("\n=== Version optimisée ===")
    vectors_opt, paths_opt = generate_vectors_optimized(
        siglip,
        images,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    print(f"Traitement terminé: {len(vectors_opt)} vecteurs générés")
