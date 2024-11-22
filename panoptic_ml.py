import os.path

from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity

from panoptic.core.plugin.plugin import APlugin
from panoptic.models import Instance, ActionContext, PropertyId
from panoptic.models.results import Group, ActionResult
from panoptic.utils import group_by_sha1
from panoptic.core.plugin.plugin_project_interface import PluginProjectInterface

from .compute.similarity import get_similar_images_from_text, get_text_vectors
from .compute import reload_tree, get_similar_images, make_clusters
from .compute_vector_task import ComputeVectorTask

import numpy as np

class PluginParams(BaseModel):
    """
    @greyscale: if this is checked, vectors can be recomputed but this time images will be converted to greyscale before
    """
    greyscale: bool = False


class PanopticML(APlugin):
    """
    Default Machine Learning plugin for Panoptic
    Uses CLIP to generate vectors and FAISS for clustering / similarity functions
    """

    def __init__(self, project: PluginProjectInterface, plugin_path: str, name: str):
        super().__init__(name=name, project=project, plugin_path=plugin_path)
        self.params: PluginParams = PluginParams()
        reload_tree(self.data_path)

        self.project.on_instance_import(self.compute_image_vector)
        self.project.on_instance_import(self.compute_image_vector_greyscale)
        self.add_action_easy(self.find_images, ['similar'])
        self.add_action_easy(self.compute_clusters, ['group'])
        self.add_action_easy(self.cluster_by_tags, ['group'])
        self.add_action_easy(self.search_by_text, ['execute'])

    async def start(self):
        await super().start()
        vectors = await self.project.get_vectors(self.name, 'clip')
        vectors_greyscale = await self.project.get_vectors(self.name, 'clip_greyscale')

        # TODO: handle this properly with an import hook
        if not os.path.exists(os.path.join(self.data_path, 'tree_faiss.pkl')) and len(vectors) > 0:
            from .compute import compute_faiss_index
            await compute_faiss_index(self.data_path, self.project, self.name, 'clip')
            reload_tree(self.data_path)

        if not os.path.exists(os.path.join(self.data_path, 'tree_faiss_greyscale.pkl')) and len(vectors_greyscale) > 0 and self.params.greyscale:
            from .compute import compute_faiss_index
            await compute_faiss_index(self.data_path, self.project, self.name, 'clip_greyscale')
            reload_tree(self.data_path)

    async def compute_image_vector(self, instance: Instance):
        task = ComputeVectorTask(self.project, self.name, 'clip', instance, self.data_path)
        self.project.add_task(task)

    async def compute_image_vector_greyscale(self, instance: Instance):
        if self.params.greyscale:
            task = ComputeVectorTask(self.project, self.name, 'clip_greyscale', instance, self.data_path, greyscale=True)
            self.project.add_task(task)
        else:
            pass

    async def compute_clusters(self, context: ActionContext, nb_clusters: int = 10, ignore_color: bool = False):
        """
        Computes images clusters with Faiss Kmeans
        @nb_clusters: requested number of clusters
        """
        instances = await self.project.get_instances(context.instance_ids)
        sha1_to_instance = group_by_sha1(instances)
        sha1_to_ahash = {i.sha1: i.ahash for i in instances}
        sha1s = list(sha1_to_instance.keys())
        if not sha1s:
            return None

        if not ignore_color:
            vectors = await self.project.get_vectors(source=self.name, vector_type='clip', sha1s=sha1s)
        else:
            vectors = await self.project.get_vectors(source=self.name, vector_type='clip_greyscale', sha1s=sha1s)
        clusters, distances = make_clusters(vectors, method="kmeans", nb_clusters=nb_clusters)
        groups = []
        for cluster, distance in zip(clusters, distances):
            group = Group(score=distance)
            group.sha1s = sorted(cluster, key=lambda sha1: sha1_to_ahash[sha1])
            groups.append(group)
        for i, g in enumerate(groups):
            g.name = f"Cluster {i}"

        return ActionResult(groups=groups)

    async def find_images(self, context: ActionContext):
        """
        :return {
          min: 0. ; images are considered highly dissimilar
          max: 1. ; images are considered identical
          metric: similarity ; Cosine similarity, compute the cosine similarity between the images vectors. See: https://en.wikipedia.org/wiki/Cosine_similarity for more.
        }
        """
        instances = await self.project.get_instances(context.instance_ids)
        sha1s = [i.sha1 for i in instances]
        ignore_sha1s = set(sha1s)
        vectors = await self.project.get_vectors(source=self.name, vector_type='clip', sha1s=sha1s)
        vector_datas = [x.data for x in vectors]
        res = get_similar_images(vector_datas)
        index = {r['sha1']: r['dist'] for r in res if r['sha1'] not in ignore_sha1s}

        res_sha1s = list(index.keys())
        res_scores = [index[sha1] for sha1 in res_sha1s]

        res = Group(sha1s=res_sha1s, scores=res_scores)
        return ActionResult(instances=res)

    async def search_by_text(self, context: ActionContext, text: str):
        context_instances = await self.project.get_instances(context.instance_ids)
        context_sha1s = [i.sha1 for i in context_instances]

        text_instances = get_similar_images_from_text(text)

        # filter out images if they are not in the current context
        filtered_instances = [inst for inst in text_instances if inst['sha1'] in context_sha1s]

        index = {r['sha1']: r['dist'] for r in filtered_instances}
        res_sha1s = list(index.keys())
        res_scores = [index[sha1] for sha1 in res_sha1s]
        res = Group(sha1s=res_sha1s, scores=res_scores)
        res.name = "Text Search: " + text
        # rename instances ?
        return ActionResult(instances=res)

    async def cluster_by_tags(self, context: ActionContext, tags: PropertyId):
        instances = await self.project.get_instances(context.instance_ids)
        sha1_to_instance = group_by_sha1(instances)
        sha1s = list(sha1_to_instance.keys())
        if not sha1s:
            return None
        # TODO: get tags text from the PropertyId
        tags_text = [t.value for t in await self.project.get_tags(property_ids=[tags])]
        text_vectors = get_text_vectors(tags_text)
        pano_vectors = await self.project.get_vectors(source=self.name, vector_type='clip', sha1s=sha1s)
        vectors, sha1s = zip(*[(i.data, i.sha1) for i in pano_vectors])
        sha1s_array = np.asarray(sha1s)
        text_vectors_reshaped = np.squeeze(text_vectors, axis=1)

        images_vectors_norm = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        text_vectors_norm = text_vectors_reshaped / np.linalg.norm(text_vectors_reshaped, axis=1, keepdims=True)

        matrix = cosine_similarity(images_vectors_norm, text_vectors_norm)
        closest_text_indices = np.argmax(matrix, axis=1)
        similarities = np.max(matrix, axis=1)

        clusters = []
        distances = []

        for text_index in list(set(closest_text_indices)):
            cluster = sha1s_array[closest_text_indices == text_index]
            cluster_sim = similarities[closest_text_indices == text_index]
            distance = (1 - np.mean(cluster_sim)) * 100
            sorting_index = cluster_sim.argsort()
            sorted_cluster = cluster[sorting_index[::-1]]
            clusters.append(sorted_cluster)
            distances.append(distance)

        groups = []
        for cluster, distance in zip(clusters, distances):
            group = Group(score=distance)
            group.sha1s = list(cluster)
            groups.append(group)
        for i, g in enumerate(groups):
            g.name = f"Cluster {tags_text[i]}"

        return ActionResult(groups=groups)

