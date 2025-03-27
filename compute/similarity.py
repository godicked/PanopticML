import numpy as np


def get_text_vectors(texts: [str], transformer):
    vectors = []
    if transformer.can_handle_text:
        for text in texts:
            vectors.append(transformer.to_text_vector(text))
    else:
        raise ValueError(f"The selected transformer {transformer.name_or_path} does not support text vectors.")
    return np.asarray(vectors)
