import io
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
import numpy as np


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

    def to_vector(self, image: Image) -> np.ndarray:
        inputs = self.processor(images=[image], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy().flatten()

    def to_text_vector(self, text: str) -> np.ndarray:
        inputs = self.processor(text=[text], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy().flatten()

class SIGLIPTransformer(AutoTransformer):
    def __init__(self):
        model_name = "openai/clip-vit-base-patch32"
        super().__init__(model_name)
        self.name = "SIGLIP"

    @property
    def can_handle_text(self):
        return True


def get_images(folder):
    return [f for f in folder.iterdir() if
            f.suffix in ['.jpg', '.jpeg', '.png', '.gif'] and f.name != 'cropped_chat.png']


def generate_vectors(transformer: Transformer, images=None):
    vectors = []
    images = get_images() if not images else images

    for img_path in tqdm(images):
        with open(img_path, mode='rb') as f:
            image = Image.open(io.BytesIO(f.read()))
            image = image.convert('RGB')
        vectors.append(transformer.to_vector(image))
    return vectors, images


if __name__ == "__main__":
    folder = r"D:\CorpusImage\documerica\extracted_images"
    siglip = SIGLIPTransformer()
    vectors, images = generate_vectors(siglip, get_images(Path(folder)))