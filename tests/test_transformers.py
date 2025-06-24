import os
from itertools import product

import pytest
import numpy as np
from PIL import Image

from compute.transformers import get_transformer, TransformerName
from models import VectorType
from utils import transform_image


def create_test_image():
    """Crée une image de test simple"""
    # Créer une image RGB de 224x224 pixels (taille standard)
    image_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(image_array)


@pytest.mark.parametrize("model_name, vector_type", list(product(TransformerName, VectorType)))
def test_transformer(model_name, vector_type):
    """Test tous les transformers disponibles"""
    for img in [f for f in os.listdir('./resources') if f.split('.')[-1] in ['jpg', 'jpeg', 'png', 'gif']]:
        with open(os.path.join('./resources', img), mode='rb') as f:
            image_data = f.read()
        test_image = transform_image(vector_type, image_data)
        test_text = "This is a test image of a cat"

        print(f"\n=== Testing {model_name.value.upper()} with image {img} ===")

        # Obtenir le transformer
        transformer = get_transformer(model_name)
        print(f"✓ Transformer {model_name} initialisé avec succès")

        # Tester la conversion d'image en vecteur
        print("Testing image to vector conversion...")
        image_vector = transformer.to_vector(test_image)

        # Vérifications
        assert isinstance(image_vector, np.ndarray), f"Le résultat doit être un numpy array pour {model_name}"
        assert image_vector.size > 0, f"Le vecteur ne doit pas être vide pour {model_name}"
        print(f"✓ Image convertie en vecteur de taille: {image_vector.shape}")

    # Tester la conversion de texte en vecteur si supportée
    if transformer.can_handle_text:
        print("Testing text to vector conversion...")
        text_vector = transformer.to_text_vector(test_text)

        # Vérifications
        assert isinstance(text_vector,
                          np.ndarray), f"Le résultat texte doit être un numpy array pour {model_name}"
        assert text_vector.size > 0, f"Le vecteur texte ne doit pas être vide pour {model_name}"
        print(f"✓ Texte converti en vecteur de taille: {text_vector.shape}")
    else:
        print("✗ Ce transformer ne supporte pas la conversion de texte")

# rajouter les tests pour comparer 3 textes à une images et voir si ça marche bien
# rajouter les tests pour générer un arbre faiss et le query