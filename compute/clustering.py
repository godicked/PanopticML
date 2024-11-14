import faiss
from scipy.stats import hmean
import numpy as np
from sklearn.metrics import silhouette_score
from panoptic.models import ComputedValue, Vector


def make_clusters(vectors: list[Vector],  **kwargs) -> (list[list[str]], list[int]):
    res_clusters = []
    res_distances = []
    vectors, sha1 = zip(*[(i.data, i.sha1) for i in vectors])
    sha1 = np.asarray(sha1)
    clusters: np.ndarray

    clusters, distances = _make_clusters_faiss(vectors, **kwargs)

    for cluster in list(set(clusters)):
        sha1_cluster = sha1[clusters == cluster]
        current_cluster_distances = distances[clusters == cluster]
        # sort current cluster by the distances
        # sorted_indices = np.argsort(current_cluster_distances)
        # sorted_cluster = sha1_cluster[sorted_indices]

        if distances is not None:
            res_distances.append(hmean(current_cluster_distances))
        res_clusters.append(list(sha1_cluster))
    # sort clusters by distances
    sorted_clusters = [cluster for _, cluster in sorted(zip(res_distances, res_clusters))]
    return sorted_clusters, sorted(res_distances)


def _make_clusters_faiss(vectors, nb_clusters=6, **kwargs) -> (np.ndarray, np.ndarray):
    def _make_single_kmean(vectors, nb_clusters):
        kmean = faiss.Kmeans(vectors.shape[1], nb_clusters, niter=20, verbose=False)
        kmean.train(vectors)
        return kmean.index.search(vectors, 1)

    vectors = np.asarray(vectors)
    if nb_clusters == -1:
        k_silhouettes = []
        max_clusters = min(len(vectors), 100)
        for k in custom_range(3, max_clusters + 1, [10, 25, 50, 75], increments=[2, 3, 4, 5]):
            distances, indices = _make_single_kmean(vectors, k)
            indices = indices.flatten()
            k_silhouettes.append(silhouette_score(vectors, indices))
        nb_clusters = int(np.argmax(k_silhouettes))
    distances, indices = _make_single_kmean(vectors, nb_clusters)
    return indices.flatten(), distances.flatten()


def custom_range(min_i, max_i, steps, increments):
    """
    Generate a range of values from min_i to max_i with a variable increment for each step
    :param min_i: first value
    :param max_i: last value
    :param steps: values for which increment should change
    :param increments: increments values
    :return:
    """
    i = min_i
    current_step = 0
    current_incr = 1
    while i < max_i:
        yield i
        if i >= steps[current_step] and current_step < len(steps) - 1:
            current_incr = increments[current_step]
            current_step += 1
        i += current_incr