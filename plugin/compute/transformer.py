import numpy as np
from PIL.Image import Image

from panoptic.models import VectorType


class Transformer(object):
    def __init__(self):
        import torch
        from transformers import logging
        logging.set_verbosity_error()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def can_handle_text(self):
        return False


class AutoTransformer(Transformer):
    def __init__(self, hugging_face_model=None):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer, AutoProcessor

        try:
            self.model = AutoModel.from_pretrained(hugging_face_model).to(self.device)
        except BaseException:
            pass

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(hugging_face_model)
        except BaseException:
            pass

        try:
            self.processor = AutoProcessor.from_pretrained(hugging_face_model)
        except BaseException:
            if self.processor is None:
                from transformers import AutoImageProcessor
                self.processor = AutoImageProcessor.from_pretrained(hugging_face_model)

        self.name = hugging_face_model

    @property
    def can_handle_text(self):
        return True

    def to_vector(self, image: Image) -> np.ndarray:
        # Preprocess the image (batch of 1)
        inputs = self.processor(images=[image], return_tensors="pt").to(self.device)

        # If the model implements get_image_features (e.g. CLIP), use it directly
        if hasattr(self.model, "get_image_features"):
            image_embeddings = self.model.get_image_features(**inputs)
            vector = image_embeddings.detach().cpu().numpy()[0]

        else:
            # Run forward pass
            outputs = self.model(**inputs)

            # Try pooler_output if exists (e.g. some BERT variants)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                pooled = outputs.pooler_output
            else:
                # Otherwise use last_hidden_state with mean pooling
                pooled = outputs.last_hidden_state.mean(dim=1)

            vector = pooled.detach().cpu().numpy()[0]

        return vector

    def to_text_vector(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(text=text, return_tensors="pt").to(self.device)
        text_embeddings = self.model.get_text_features(**inputs)
        # Convertir les embeddings en tableau numpy
        embedding_as_np = text_embeddings.cpu().detach().numpy()
        return embedding_as_np.reshape(1, -1)


class TransformerManager:
    def __init__(self):
        self.transformers: dict[int, AutoTransformer] = {}

    def get(self, vec_type: VectorType):
        if self.transformers.get(vec_type.id):
            return self.transformers[vec_type.id]
        self.transformers[vec_type.id] = AutoTransformer(vec_type.params["model"])
        return self.transformers[vec_type.id]
