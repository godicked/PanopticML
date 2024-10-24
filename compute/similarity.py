from __future__ import annotations

import os
import pickle

import faiss
import numpy as np
from panoptic.models import ComputedValue, Vector
from panoptic.core.plugin.plugin_project_interface import PluginProjectInterface

from .utils import load_similarity_tree, TRANSFORMER


def reload_tree(path: str):
    global SIMILARITY_TREE
    SIMILARITY_TREE = load_similarity_tree(path)

class SimilarityFaissWithLabel:
    def __init__(self, images: list[Vector]):
        vectors, sha1_list = zip(*[(i.data, i.sha1) for i in images])
        vectors = np.asarray(vectors)
        faiss.normalize_L2(vectors)
        self.image_labels = sha1_list
        # create the faiss index based on this post: https://anttihavanko.medium.com/building-image-search-with-openai-clip-5a1deaa7a6e2
        vector_size = vectors.shape[1]
        index = faiss.IndexFlatIP(vector_size)
        self.tree = index
        self.tree.add(np.asarray(vectors))

    def query(self, image: np.ndarray, k=99999):
        # by normalizing it allows to search by cosine distance instead of inner product, need to do text / image sim
        faiss.normalize_L2(image)
        vector = image.reshape(1, -1)
        dist, ind = self.tree.search(vector, k)
        indices = [x for x in ind[0]]
        distances = [x if x <= 1.0 else 0 for x in dist[0]]  # avoid some strange overflow behavior
        return [{'sha1': self.image_labels[i], 'dist': float('%.2f' % (distances[index]))} for index, i in
                enumerate(indices)]


SIMILARITY_TREE: SimilarityFaissWithLabel | None = None


def create_similarity_tree_faiss(path: str, images: list[Vector]):
    tree = SimilarityFaissWithLabel(images)
    with open(os.path.join(path, 'tree_faiss.pkl'), 'wb') as f:
        pickle.dump(tree, f)
    global SIMILARITY_TREE
    SIMILARITY_TREE = tree


async def compute_faiss_index(path: str, db: PluginProjectInterface, source: str, type_: str):
    vectors = await db.get_vectors(source, type_)
    create_similarity_tree_faiss(path, vectors)


async def get_similar_images_from_text(input_text: str):
    if TRANSFORMER.can_handle_text:
        vec = TRANSFORMER.to_text_vector(input_text)
        return SIMILARITY_TREE.query(vec)


def get_similar_images(vectors: list[np.ndarray]):
    if not SIMILARITY_TREE:
        raise ValueError("Cannot compute image similarity since KDTree was not computed yet")
    vector = np.mean(vectors, axis=0)
    return SIMILARITY_TREE.query(np.asarray([vector]))
