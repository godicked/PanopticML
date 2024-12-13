from random import randrange
from typing import Dict

import numpy as np
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity

from panoptic.core.plugin.plugin import APlugin
from panoptic.core.plugin.plugin_project_interface import PluginProjectInterface
from panoptic.models import Instance, ActionContext, PropertyId
from panoptic.models.results import Group, ActionResult, Notif, NotifType, NotifFunction, ScoreList, Score
from panoptic.utils import group_by_sha1
from .compute import make_clusters
from .compute.faiss_tree import load_faiss_tree, create_faiss_tree, FaissTree
from .compute.similarity import get_text_vectors
from .compute_vector_task import ComputeVectorTask
from .models import VectorType


class PluginParams(BaseModel):
    """
    @greyscale: if this is checked, vectors can be recomputed but this time images will be converted to greyscale before
    """
    similarity_vector: VectorType = VectorType.clip


class PanopticML(APlugin):
    """
    Default Machine Learning plugin for Panoptic
    Uses CLIP to generate vectors and FAISS for clustering / similarity functions
    """

    def __init__(self, project: PluginProjectInterface, plugin_path: str, name: str):
        super().__init__(name=name, project=project, plugin_path=plugin_path)
        self.params: PluginParams = PluginParams()

        self.project.on_instance_import(self.compute_image_vector)
        self.add_action_easy(self.find_images, ['similar'])
        self.add_action_easy(self.compute_clusters, ['group'])
        self.add_action_easy(self.cluster_by_tags, ['group'])
        self.add_action_easy(self.search_by_text, ['execute'])
        self._comp_vec_desc = self.add_action_easy(self.compute_vectors, ['execute'])
        self._comp_all_vec_desc = self.add_action_easy(self.compute_all_vectors, ['execute'])

        self.trees: Dict[VectorType, FaissTree] = {}

    async def start(self):
        await super().start()

        [await self.get_tree(t) for t in VectorType]

    def _get_vector_func_notifs(self, vec_type: VectorType):
        res = [
            NotifFunction(self._comp_vec_desc.id,
                          ActionContext(ui_inputs={"vec_type": vec_type.value}),
                          message=f"Compute all vectors of type {vec_type.value}"),
            NotifFunction(self._comp_all_vec_desc.id,
                          ActionContext(),
                          message="Compute vectors off all types")
        ]
        return res

    async def compute_vectors(self, context: ActionContext, vec_type: VectorType):
        instances = await self.project.get_instances(ids=context.instance_ids)
        for i in instances:
            await self.compute_image_vector(i, vec_type)

        notif = Notif(type=NotifType.INFO,
                      name="ComputeVector",
                      message=f"Successfuly started compute of vectors of type {vec_type.value}")
        return ActionResult(notifs=[notif])

    async def compute_all_vectors(self, context: ActionContext):
        res = [await self.compute_vectors(context, t) for t in VectorType]
        return ActionResult(notifs=[n for r in res for n in r.notifs])

    async def compute_image_vector(self, instance: Instance, vector: VectorType):
        task = ComputeVectorTask(self, self.name, vector, instance, self.data_path)
        self.project.add_task(task)

    async def compute_clusters(self, context: ActionContext, vec_type: VectorType, nb_clusters: int = 10):
        instances = await self.project.get_instances(context.instance_ids)
        sha1_to_instance = group_by_sha1(instances)
        sha1_to_ahash = {i.sha1: i.ahash for i in instances}
        sha1s = list(sha1_to_instance.keys())

        if not sha1s:
            empty_notif = Notif(NotifType.ERROR, name="NoData", message="No instance found")
            return ActionResult(notifs=[empty_notif])

        vectors = await self.project.get_vectors(source=self.name, vector_type=vec_type.value, sha1s=sha1s)

        if not vectors:
            empty_notif = Notif(NotifType.ERROR,
                                name="NoData",
                                message=f"""For the clustering function image vectors are needed.
                                        No such vectors ({vec_type.value}) could be found. 
                                        Compute the vectors and try again.) """,
                                functions=self._get_vector_func_notifs(vec_type))
            return ActionResult(notifs=[empty_notif])
        clusters, distances = make_clusters(vectors, method="kmeans", nb_clusters=nb_clusters)
        groups = []
        for cluster, distance in zip(clusters, distances):
            group = Group(score=Score(min=0, max=100, max_is_best=False, value=distance))
            group.sha1s = sorted(cluster, key=lambda sha1: sha1_to_ahash[sha1])
            groups.append(group)
        for i, g in enumerate(groups):
            g.name = f"Cluster {i}"

        return ActionResult(groups=groups)

    async def find_images(self, context: ActionContext, vec_type: VectorType = VectorType.clip):
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
        vectors = await self.project.get_vectors(source=self.name, vector_type=vec_type.value, sha1s=sha1s)

        if not vectors:
            return ActionResult(notifs=[Notif(
                NotifType.ERROR,
                name="NoData",
                message=f"""For the similarity function image vectors are needed.
                            No such vectors ({vec_type.value}) could be found. 
                            Compute the vectors and try again.) """,
                functions=self._get_vector_func_notifs(vec_type))])

        vector_datas = [x.data for x in vectors]

        tree = await self.get_tree(vec_type)
        if not tree:
            notif = Notif(type=NotifType.ERROR, name="NoFaissTree",
                          message=f"No Faiss tree could be loaded for vec_type {vec_type.value}")
            return ActionResult(notifs=[notif])

        res = tree.query(vector_datas)
        index = {r['sha1']: r['dist'] for r in res if r['sha1'] not in ignore_sha1s}

        res_sha1s = list(index.keys())
        res_scores = ScoreList(min=0, max=1, values=[index[sha1] for sha1 in res_sha1s],
                               max_is_best=True,
                               description="Similarity between 0 and 1. 1 is best")

        res = Group(sha1s=res_sha1s, scores=res_scores)
        return ActionResult(groups=[res])

    async def search_by_text(self, context: ActionContext, vec_type: VectorType = VectorType.clip, text: str = ''):
        if text == '':
            notif = Notif(type=NotifType.ERROR, name="EmptySearchText",
                          message="Please give a valid and not empty text search argument")
            return ActionResult(notifs=[notif])

        context_instances = await self.project.get_instances(context.instance_ids)
        context_sha1s = [i.sha1 for i in context_instances]

        tree = await self.get_tree(vec_type)
        if not tree:
            notif = Notif(type=NotifType.ERROR, name="NoFaissTree",
                          message=f"No Faiss tree could be loaded for vec_type {vec_type.value}")
            return ActionResult(notifs=[notif])

        text_instances = tree.query_texts([text])

        # filter out images if they are not in the current context
        filtered_instances = [inst for inst in text_instances if inst['sha1'] in context_sha1s]

        index = {r['sha1']: r['dist'] for r in filtered_instances}
        res_sha1s = list(index.keys())
        res_scores = [index[sha1] for sha1 in res_sha1s]
        scores = ScoreList(min=0, max=1, values=res_scores)
        res = Group(sha1s=res_sha1s, scores=scores)
        res.name = "Text Search: " + text
        return ActionResult(groups=[res])

    async def cluster_by_tags(self, context: ActionContext, tags: PropertyId, vec_type: VectorType = VectorType.clip):
        instances = await self.project.get_instances(context.instance_ids)
        sha1_to_instance = group_by_sha1(instances)
        sha1s = list(sha1_to_instance.keys())
        if not sha1s:
            return None
        # TODO: get tags text from the PropertyId
        tags_text = [t.value for t in await self.project.get_tags(property_ids=[tags])]
        text_vectors = get_text_vectors(tags_text)
        pano_vectors = await self.project.get_vectors(source=self.name, vector_type=vec_type.value, sha1s=sha1s)
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

    async def get_tree(self, vec_type: VectorType):
        tree = self.trees.get(vec_type)
        if tree:
            return tree
        tree = load_faiss_tree(self, vec_type)
        if tree:
            self.trees[vec_type] = tree
            return tree
        tree = await create_faiss_tree(self, vec_type)
        if tree:
            self.trees[vec_type] = tree
            return tree

    async def update_tree(self, vec_type: VectorType):
        tree = await create_faiss_tree(self, vec_type)
        self.trees[vec_type] = tree
        print(f"updated {vec_type.value} faiss tree")
        return tree
