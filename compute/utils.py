import os
import pickle

from .transformers import get_transformer

TRANSFORMER = get_transformer('clip')
def load_similarity_tree(path: str):
    path = os.path.join(path, 'tree_faiss.pkl')
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)
