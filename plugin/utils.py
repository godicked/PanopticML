import io

from PIL import Image


def preprocess_image(image_data: bytes, params: dict):
    image = Image.open(io.BytesIO(image_data))
    if params.get('greyscale'):
        image = image.convert('L').convert('RGB')
    else:
        image = image.convert('RGB')
    return image
