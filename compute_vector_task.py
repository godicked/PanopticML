from __future__ import annotations

import io
import logging
from typing import TYPE_CHECKING

from .models import VectorType

if TYPE_CHECKING:
    from . import PanopticML

import aiofiles
from PIL import Image

from panoptic.core.task.task import Task
from panoptic.models import Instance, Vector

from .compute.utils import TRANSFORMER

logger = logging.getLogger('PanopticML:VectorTask')


class ComputeVectorTask(Task):
    def __init__(self, plugin: PanopticML, source: str, type_: VectorType, instance: Instance,
                 data_path: str):
        super().__init__()
        self.plugin = plugin
        self.project = plugin.project
        self.source = source
        self.type = type_
        self.instance = instance
        self.name = f'Clip Vectors ({type_.value})'
        self.data_path = data_path
        self.key += f"_{type_.value}"

    async def run(self):
        instance = self.instance
        exist = await self.project.vector_exist(self.source, self.type.value, instance.sha1)
        if exist:
            return

        image_data = await self.project.get_project().db.get_large_image(instance.sha1)
        if not image_data:
            file = instance.url
            async with aiofiles.open(file, mode='rb') as file:
                image_data = await file.read()

        vector_data = None
        if self.type == VectorType.clip:
            vector_data = await self._async(self.compute_image_clip, image_data)
        if self.type == VectorType.clip_grey:
            vector_data = await self._async(self.compute_image_clip_gray, image_data)
        if vector_data is None:
            return
        vector = Vector(self.source, self.type.name, instance.sha1, vector_data)
        res = await self.project.add_vector(vector)
        del vector
        return res

    async def run_if_last(self):
        await self.plugin.update_tree(self.type)

    @staticmethod
    def compute_image_clip(image_data: bytes):
        image = Image.open(io.BytesIO(image_data))
        image = image.convert('RGBA')
        vector = TRANSFORMER.to_vector(image)

        del image
        return vector

    @staticmethod
    def compute_image_clip_gray(image_data: bytes):
        image = Image.open(io.BytesIO(image_data))
        image = image.convert('L')
        vector = TRANSFORMER.to_vector(image)

        del image
        return vector
