import faiss
from scipy.stats import hmean
import numpy as np
from panoptic.models import Vector


def make_clusters(vectors: list[Vector], **kwargs) -> (list[list[str]], list[int]):
    res_clusters = []
    res_distances = []
    vectors, sha1 = zip(*[(i.data, i.sha1) for i in vectors])
    sha1 = np.asarray(sha1)
    clusters: np.ndarray

    clusters, distances = _make_clusters_faiss(vectors, **kwargs)

    for cluster in list(set(clusters)):
        sha1_cluster = sha1[clusters == cluster]
        current_cluster_distances = distances[clusters == cluster]
        if distances is not None:
            res_distances.append(np.mean(current_cluster_distances))
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
        import hdbscan
        clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
        clusterer.fit(vectors)
        indices = clusterer.labels_
        probabilities = clusterer.probabilities_
        distances = np.zeros_like(probabilities, dtype=np.float32)
        unique_clusters = np.unique(indices)
        # compute distances just like the one returned by kmeans to have consistent metrics
        for cluster_id in unique_clusters:
            if cluster_id == -1:
                distances[indices == -1] = 100.0
                continue
            cluster_mask = (indices == cluster_id)
            cluster_vectors = vectors[cluster_mask]
            cluster_probabilities = probabilities[cluster_mask]
            center_local_index = np.argmax(cluster_probabilities)
            center_vector = cluster_vectors[center_local_index].reshape(1, -1)
            dists = faiss.pairwise_distances(center_vector, cluster_vectors)[0]
            distances[cluster_mask] = dists
    else:
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
