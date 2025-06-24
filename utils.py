import io

from PIL import Image

from models import VectorType


def transform_image(vector_type: VectorType, image_data: bytes):
    image = Image.open(io.BytesIO(image_data))
    if vector_type == VectorType.rgb:
        image = image.convert('RGB')
    else:
        image = image.convert('L').convert('RGB')
    return image