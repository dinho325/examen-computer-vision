from datasets import load_dataset
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

def prediction():
    # Charger le modele entrainé
    model = tf.keras.models.load_model("model_fruits360.h5")
    # Charger une nouvelle image
    image_path = image
    image = Image.open(image_path)

    # Appliquer les transformations (resize + normalisation)
    image = image.resize((100, 100))  
    image = np.array(image) / 255.0    
    image = np.expand_dims(image, axis=0)  

    # Faire la prédiction
    y_pred_probs = model.predict(image)  
    y_pred_class = np.argmax(y_pred_probs, axis=1)[0]  

    # Afficher l’image et la classe prédite
    plt.imshow(Image.open(image_path))
    plt.axis("off")
    plt.title(f"Prédiction : Classe {y_pred_class}")
    plt.show()
