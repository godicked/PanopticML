import numpy as np

from .utils import TRANSFORMER


def get_text_vectors(texts: [str]):
    vectors = []
    if TRANSFORMER.can_handle_text:
        for text in texts:
            vectors.append(TRANSFORMER.to_text_vector(text))
    return np.asarray(vectors)
